"""
sample_ga.py
Genetic Algorithm (GA) based molecular latent vector generator

Uses DEAP library to generate low-energy molecular representations in binary latent space via evolutionary strategies.

Core strategies:
1. Initialization: Random binary population generation
2. Selection: Tournament Selection
3. Crossover: Uniform Crossover
4. Mutation: Bit-flip Mutation
5. Elitism: Preserve best individuals per generation
"""

import argparse
import numpy as np
import h5py
import logging
import os
from pathlib import Path
from tqdm import tqdm

from deap import base, creator, tools, algorithms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class GAMoleculeGenerator:
    """GA-based molecular latent vector generator"""

    def __init__(
        self,
        qubo_matrix: np.ndarray,
        n_dim: int = 128,
        pop_size: int = 100,
        n_generations: int = 200,
        cx_prob: float = 0.7,
        mut_prob: float = 0.1,
        mut_indpb: float = 0.05,
        tournsize: int = 3,
        elite_size: int = 5,
        random_seed: int = None
    ):
        """
        Initialize GA generator

        Args:
            qubo_matrix: QUBO matrix for energy (fitness) calculation
            n_dim: Latent vector dimension
            pop_size: Population size
            n_generations: Number of generations
            cx_prob: Crossover概率
            mut_prob: Mutation概率
            mut_indpb: 每个位的独立Mutation概率
            tournsize: Tournament selection size
            elite_size: Elite preservation数量
            random_seed: Random seed
        """
        self.qubo_matrix = qubo_matrix
        self.n_dim = n_dim
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.mut_indpb = mut_indpb
        self.tournsize = tournsize
        self.elite_size = elite_size

        if random_seed is not None:
            np.random.seed(random_seed)

        self._setup_deap()

    def _setup_deap(self):
        """Configure DEAP framework"""
        # Clean up possible old definitions
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Define fitness (minimize energy)
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        # Register individual and population generators
        self.toolbox.register("attr_bit", np.random.randint, 0, 2)
        self.toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,
            self.toolbox.attr_bit,
            n=self.n_dim
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )

        # Register genetic operations
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=self.mut_indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

    def _evaluate(self, individual):
        """Calculate individual fitness (QUBO energy)"""
        x = np.array(individual, dtype=np.float64)
        energy = x @ self.qubo_matrix @ x
        return (energy,)

    def generate(self, n_samples: int, collect_all_generations: bool = True):
        """
        Run genetic algorithm to generate samples

        Args:
            n_samples: Number of samples to generate
            collect_all_generations: Whether to collect individuals from all generations

        Returns:
            samples: 生成的二进制向量 (n_samples, n_dim)
            energies: 对应的能量值
            stats: Statistics
        """
        log.info(f"Starting GA generation: pop_size={self.pop_size}, "
                 f"generations={self.n_generations}")

        # Initialize population
        pop = self.toolbox.population(n=self.pop_size)

        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Collect all individuals
        all_individuals = []
        all_energies = []

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("min", np.min)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("std", np.std)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals", "min", "avg", "max", "std"]

        # Record initial generation
        record = stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)

        if collect_all_generations:
            for ind in pop:
                all_individuals.append(list(ind))
                all_energies.append(ind.fitness.values[0])

        # Evolution loop
        for gen in tqdm(range(1, self.n_generations + 1), desc="GA Evolution"):
            # Elite preservation
            elites = tools.selBest(pop, self.elite_size)
            elites = [self.toolbox.clone(e) for e in elites]

            # Select next generation
            offspring = self.toolbox.select(pop, len(pop) - self.elite_size)
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Crossover
            for i in range(0, len(offspring) - 1, 2):
                if np.random.random() < self.cx_prob:
                    self.toolbox.mate(offspring[i], offspring[i + 1])
                    del offspring[i].fitness.values
                    del offspring[i + 1].fitness.values

            # Mutation
            for mutant in offspring:
                if np.random.random() < self.mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Merge elites and offspring
            pop[:] = elites + offspring

            # Record statistics
            record = stats.compile(pop)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # Collect individuals
            if collect_all_generations:
                for ind in pop:
                    all_individuals.append(list(ind))
                    all_energies.append(ind.fitness.values[0])

        # Print final statistics
        log.info("\nGA Evolution Summary:")
        log.info(f"  Initial best energy: {logbook[0]['min']:.6f}")
        log.info(f"  Final best energy: {logbook[-1]['min']:.6f}")
        log.info(f"  Improvement: {logbook[0]['min'] - logbook[-1]['min']:.6f}")

        # Select top n_samples samples
        all_individuals = np.array(all_individuals, dtype=np.int8)
        all_energies = np.array(all_energies, dtype=np.float32)

        # Deduplicate
        unique_individuals, unique_indices = np.unique(
            all_individuals, axis=0, return_index=True
        )
        unique_energies = all_energies[unique_indices]

        log.info(f"  Total unique samples: {len(unique_individuals)}")

        # Sort by energy, select best
        sorted_indices = np.argsort(unique_energies)[:n_samples]
        samples = unique_individuals[sorted_indices]
        energies = unique_energies[sorted_indices]

        return samples, energies, logbook


def main():
    parser = argparse.ArgumentParser(
        description="Generate molecular latent vectors using Genetic Algorithm",
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

    # GA parameters
    parser.add_argument("--n_samples", type=int, default=10000,
                        help="Number of samples to generate")
    parser.add_argument("--n_dim", type=int, default=128,
                        help="Dimension of latent vectors")
    parser.add_argument("--pop_size", type=int, default=200,
                        help="Population size")
    parser.add_argument("--n_generations", type=int, default=500,
                        help="Number of generations")
    parser.add_argument("--cx_prob", type=float, default=0.7,
                        help="Crossover probability")
    parser.add_argument("--mut_prob", type=float, default=0.2,
                        help="Mutation probability")
    parser.add_argument("--mut_indpb", type=float, default=0.05,
                        help="Independent bit mutation probability")
    parser.add_argument("--tournsize", type=int, default=3,
                        help="Tournament selection size")
    parser.add_argument("--elite_size", type=int, default=10,
                        help="Number of elites to preserve")
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
    generator = GAMoleculeGenerator(
        qubo_matrix=qubo_matrix,
        n_dim=args.n_dim,
        pop_size=args.pop_size,
        n_generations=args.n_generations,
        cx_prob=args.cx_prob,
        mut_prob=args.mut_prob,
        mut_indpb=args.mut_indpb,
        tournsize=args.tournsize,
        elite_size=args.elite_size,
        random_seed=args.seed
    )

    # Generate samples
    samples, energies, logbook = generator.generate(n_samples=args.n_samples)

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
        hf.attrs['method'] = 'GA'
        hf.attrs['pop_size'] = args.pop_size
        hf.attrs['n_generations'] = args.n_generations

    log.info("✅ GA generation completed!")


if __name__ == "__main__":
    main()
