"""
sample_tpe.py
TPE (Tree-structured Parzen Estimator) based molecular latent vector generator

Uses Optuna for Bayesian optimization, guiding generation of low-energy molecular representations in binary latent space via surrogate models.

Core strategies:
1. TPE modeling: Build separate probability density estimates for good/bad samples
2. Acquisition function: Expected Improvement (EI)
3. Native support for discrete binary variables, no continuous relaxation needed
"""

import argparse
import numpy as np
import h5py
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import optuna
from optuna.samplers import TPESampler

# Disable Optuna default logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class TPEMoleculeGenerator:
    """TPE Bayesian optimization based molecular latent vector generator"""

    def __init__(
        self,
        qubo_matrix: np.ndarray,
        n_dim: int = 128,
        n_startup_trials: int = 100,
        n_ei_candidates: int = 24,
        random_seed: int = None
    ):
        """
        Initialize TPE generator

        Args:
            qubo_matrix: QUBO matrix for energy calculation
            n_dim: Latent vector dimension
            n_startup_trials: Number of random startup trials
            n_ei_candidates: Number of EI acquisition function candidates
            random_seed: Random seed
        """
        self.qubo_matrix = qubo_matrix
        self.n_dim = n_dim
        self.n_startup_trials = n_startup_trials
        self.n_ei_candidates = n_ei_candidates
        self.random_seed = random_seed

        # Store all sampled solutions
        self.all_solutions: List[np.ndarray] = []
        self.all_energies: List[float] = []

    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective: generate binary vector and calculate energy"""
        # Generate n_dim dimensional binary vector
        x = np.array([
            trial.suggest_int(f"x{i}", 0, 1)
            for i in range(self.n_dim)
        ], dtype=np.float64)

        # Calculate QUBO energy
        energy = float(x @ self.qubo_matrix @ x)

        # Store solution
        self.all_solutions.append(x.astype(np.int8))
        self.all_energies.append(energy)

        return energy

    def generate(self, n_trials: int, show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Run TPE optimization to generate samples

        Args:
            n_trials: Total number of trials (samples to generate)
            show_progress: Whether to show progress bar

        Returns:
            samples: 生成的二进制向量 (n_samples, n_dim)
            energies: 对应的能量值
            stats: 统计信息
        """
        log.info(f"Starting TPE generation: n_trials={n_trials}, "
                 f"n_startup={self.n_startup_trials}")

        # Clear previous results
        self.all_solutions = []
        self.all_energies = []

        # Create TPE sampler
        sampler = TPESampler(
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=self.n_ei_candidates,
            seed=self.random_seed
        )

        # Create study
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler
        )

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=show_progress
        )

        # Collect statistics
        stats = {
            "best_energy": study.best_value,
            "best_trial": study.best_trial.number,
            "n_trials": len(study.trials)
        }

        log.info(f"\nTPE Optimization Summary:")
        log.info(f"  Best energy: {stats['best_energy']:.6f}")
        log.info(f"  Best trial: {stats['best_trial']}")
        log.info(f"  Total trials: {stats['n_trials']}")

        # Convert to arrays
        samples = np.array(self.all_solutions, dtype=np.int8)
        energies = np.array(self.all_energies, dtype=np.float32)

        return samples, energies, stats

    def generate_sorted(
        self,
        n_trials: int,
        n_samples: int = None,
        show_progress: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate samples and return sorted by energy

        Args:
            n_trials: Total number of trials
            n_samples: Number of samples to return (default: all)
            show_progress: Whether to show progress bar

        Returns:
            samples: Sort by energy的二进制向量
            energies: 对应的能量值
            stats: 统计信息
        """
        samples, energies, stats = self.generate(n_trials, show_progress)

        # Deduplicate
        unique_samples, unique_indices = np.unique(samples, axis=0, return_index=True)
        unique_energies = energies[unique_indices]

        log.info(f"  Unique samples: {len(unique_samples)}")

        # Sort by energy
        sorted_indices = np.argsort(unique_energies)
        sorted_samples = unique_samples[sorted_indices]
        sorted_energies = unique_energies[sorted_indices]

        # Limit return count
        if n_samples is not None and n_samples < len(sorted_samples):
            sorted_samples = sorted_samples[:n_samples]
            sorted_energies = sorted_energies[:n_samples]

        return sorted_samples, sorted_energies, stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate molecular latent vectors using TPE (Bayesian Optimization)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core parameters
    parser.add_argument(
        "--qubo_csv", type=str, required=True,
        help="Path to QUBO matrix CSV file"
    )
    parser.add_argument(
        "--output_h5", type=str, required=True,
        help="Path to output H5 file"
    )

    # TPE parameters
    parser.add_argument("--n_trials", type=int, default=10000,
                        help="Number of trials (samples to generate)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of best samples to save (default: all)")
    parser.add_argument("--n_dim", type=int, default=128,
                        help="Dimension of latent vectors")
    parser.add_argument("--n_startup", type=int, default=100,
                        help="Number of random startup trials")
    parser.add_argument("--n_ei_candidates", type=int, default=24,
                        help="Number of EI candidates")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--dataset_name", type=str, default="latent_codes",
                        help="Dataset name in H5 file")

    args = parser.parse_args()

    # Load QUBO matrix
    log.info(f"Loading QUBO matrix from: {args.qubo_csv}")
    qubo_matrix = np.loadtxt(args.qubo_csv, delimiter=",")
    log.info(f"QUBO matrix shape: {qubo_matrix.shape}")

    # Create generator
    generator = TPEMoleculeGenerator(
        qubo_matrix=qubo_matrix,
        n_dim=args.n_dim,
        n_startup_trials=args.n_startup,
        n_ei_candidates=args.n_ei_candidates,
        random_seed=args.seed
    )

    # Generate samples
    samples, energies, stats = generator.generate_sorted(
        n_trials=args.n_trials,
        n_samples=args.n_samples
    )

    log.info(f"\nGenerated {len(samples)} samples")
    log.info(f"Energy range: [{energies.min():.6f}, {energies.max():.6f}]")
    log.info(f"Mean energy: {energies.mean():.6f}")

    # Save results
    output_path = Path(args.output_h5)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving results to: {output_path}")
    with h5py.File(output_path, 'w') as hf:
        hf.create_dataset(args.dataset_name, data=samples, compression="gzip")
        hf.create_dataset("energies", data=energies, compression="gzip")
        hf.attrs['method'] = 'TPE'
        hf.attrs['n_trials'] = args.n_trials
        hf.attrs['n_startup'] = args.n_startup

    log.info("✅ TPE generation completed!")


if __name__ == "__main__":
    main()
