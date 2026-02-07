# Janus-QUBO: A Duality-Aware Framework for Navigating Chemical Space with a Tunable Quantum-Inspired Landscape

A QUBO (Quadratic Unconstrained Binary Optimization) based molecular design and optimization system for drug discovery and molecular property optimization.

## Overview

Janus-QUBO combines deep learning and quantum-inspired optimization techniques to enable efficient molecular design in binary latent spaces. The framework consists of three main components:

- **Block_bAE**: Binary autoencoder for bidirectional SMILES ↔ binary latent vector conversion, using SELFIES as intermediate representation
- **Qmol_FM**: Surrogate model training (FM/MLP/FT-Transformer), QUBO formulation, molecular generation (GA/TPE) and local optimization
- **RBM**: Restricted Boltzmann Machine for molecular distribution learning and sampling

## Key Features

- **Binary Latent Space**: Discrete optimization-friendly molecular representation
- **QUBO/HUBO Optimization**: Quantum-inspired optimization for molecular properties
- **Multi-Objective Design**: Simultaneous optimization of multiple molecular properties
- **Scalable Architecture**: Parallel processing support for large-scale molecular screening

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Janus-QUBO.git
cd Janus-QUBO

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Encode SMILES to Latent Vectors

```bash
# From a plain text file (one SMILES per line)
python Block_bAE/encode_smiles.py \
    --ckpt_path path/to/model.ckpt \
    --vocab_path Block_bAE/vocab/vocabulary.json \
    --input_smiles_file input_smiles.txt \
    --output_h5_path latent_codes.h5

# From a TSV file (with pre-tokenized data)
python Block_bAE/inference_latent.py \
    --tsv_path input_molecules.tsv \
    --trained_model_ckpt_path path/to/model.ckpt \
    --output_path latent_codes.h5
```

### 2a. Generate Molecules from QUBO (de novo)

```bash
# Genetic Algorithm-based generation
python Qmol_FM/sample_ga.py \
    --qubo_csv Qmol_FM/example_data/qubo_128.csv \
    --output_h5 generated_codes.h5 \
    --n_samples 1000 --pop_size 500 --n_generations 200

# TPE (Bayesian optimization) generation
python Qmol_FM/sample_tpe.py \
    --qubo_csv Qmol_FM/example_data/qubo_128.csv \
    --output_h5 generated_codes.h5 \
    --n_trials 5000 --n_samples 1000

# Kaiwu quantum annealing (requires Kaiwu SDK credentials)
# sample.py takes an Ising-format matrix; use kw.core.qubo_matrix_to_ising_matrix()
# to convert QUBO → Ising first, then solve:
python Qmol_FM/sample.py \
    --ising_csv ising_matrix.csv \
    --output_h5 solutions.h5 \
    --user_id YOUR_USER_ID --sdk_code YOUR_SDK_CODE \
    --num_solutions 500

# Convert Ising solutions (-1/+1) back to binary latent codes (0/1)
python Qmol_FM/convert_ising_to_qubo.py \
    --input_csv ising_solutions.csv \
    --output_h5 latent_codes_from_ising.h5
```

### 2b. Optimize Existing Latent Vectors (local search)

```bash
python Qmol_FM/opt.py \
    --qubo_csv Qmol_FM/example_data/qubo_128.csv \
    --input_h5 latent_codes.h5 \
    --output_h5 optimized_codes.h5 \
    --single_features_json single_features_detailed.json \
    --pair_configs_json pair_value_configurations.json \
    --n_jobs 8
```

### 3. Decode to SMILES

```bash
python Block_bAE/latent_to_smiles.py \
    --input_h5 optimized_codes.h5 \
    --model_ckpt_path path/to/model.ckpt \
    --vocab_path Block_bAE/vocab/vocabulary.json \
    --output_h5 output_molecules.h5
```

## Architecture

### Block_bAE (Binary Autoencoder)

- **Encoder**: GRU or Transformer-based encoder with Gumbel-Softmax binarization
- **Latent Space**: Binary vectors (128-bit default) via straight-through Gumbel-Softmax
- **Decoder**: Transformer decoder with cross-attention over latent memory
- **Training**: Reconstruction loss + KL divergence with free-bits, scheduled sampling, word dropout

### Qmol_FM (Surrogate Modeling, Generation & Optimization)

- **Surrogate Models**: FM, MLP, and FT-Transformer architectures for property prediction from binary latent codes
- **QUBO Formulation**: Converts trained FM to quadratic/higher-order unconstrained binary optimization
- **De Novo Generation**: Generate novel molecules by minimizing QUBO energy
  - Genetic Algorithm (DEAP-based) with configurable population and generations
  - TPE Bayesian optimization (Optuna-based) for efficient search
  - Quantum annealing via Kaiwu SDK
- **Local Optimization**: Improve existing molecules via bit-flip strategies
  - Single-bit flip optimization
  - Pairwise configuration optimization
  - Greedy cumulative optimization
- **Parallel Processing**: Multi-core support for batch optimization

### RBM (Sampling Module)

- **Gibbs Sampling**: Generate novel molecular latent codes
- **Batch Processing**: Memory-efficient sampling for large-scale generation
- **Energy-Based Model**: Learn molecular distribution from training set

## Project Structure

```
Janus-QUBO/
├── Block_bAE/                          # Molecular autoencoder
│   ├── model_gru_transformer.py        # Model architecture (GRU/Transformer encoder + Transformer decoder)
│   ├── train_gruencoder_transformerdecoder.py  # Training script (PyTorch Lightning)
│   ├── datamodule.py                   # Data loader
│   ├── encode_smiles.py               # SMILES → Latent encoding (from text file)
│   ├── inference_latent.py             # SMILES → Latent encoding (from TSV file)
│   ├── latent_to_smiles.py             # Latent → SMILES decoding
│   ├── calculate_properties.py         # RDKit property calculation for HDF5 datasets
│   └── vocab/
│       └── vocabulary.json             # SELFIES token vocabulary (38 tokens)
├── Qmol_FM/                            # Surrogate modeling, generation & optimization
│   ├── train_Qmol_FM.py               # Surrogate model training (FM/MLP/FT-Transformer)
│   ├── convert_fm_to_qubo.py           # FM → QUBO matrix conversion + verification
│   ├── sample_ga.py                    # GA-based molecular generation from QUBO
│   ├── sample_tpe.py                   # TPE Bayesian optimization generation from QUBO
│   ├── sample.py                       # Kaiwu quantum annealing solver interface
│   ├── opt.py                          # Local bit-flip optimizer for existing molecules
│   ├── convert_ising_to_qubo.py        # Ising → QUBO solution conversion
│   ├── utils_Qmol_FM.py               # Shared utilities (VAE wrapper, SMILES reconstructor)
│   ├── hubo.py                         # Higher-order QUBO with constraint support
│   └── example_data/
│       └── qubo_128.csv               # Pre-trained FM QUBO matrix (128-bit, MSE loss)
├── RBM/                                # Sampling module
│   ├── train_rbm.py                    # RBM training
│   └── sample_from_rbm.py             # Gibbs sampling for molecular generation
├── .gitignore
├── README.md
└── requirements.txt
```

## Dependencies

- **Deep Learning**: PyTorch ≥ 2.0, PyTorch Lightning ≥ 2.0
- **Molecular Processing**: RDKit, SELFIES
- **Data Processing**: NumPy, Pandas, H5PY
- **Machine Learning**: scikit-learn, LightGBM
- **Optimization**: Kaiwu SDK (optional, for quantum annealing hardware)

See `requirements.txt` for complete dependency list.

## Usage Notes

### Trained Models Required

This repository contains **model architecture and utility code**. To run inference or optimization, you will need:

1. **Pre-trained autoencoder checkpoint** (`.ckpt` file)
2. **Training dataset** (for FM training and RBM)

The vocabulary file is included at `Block_bAE/vocab/vocabulary.json`.

### Quantum Solver (Kaiwu SDK)

The Kaiwu quantum annealing solver operates on Ising-format matrices. The workflow is:

1. **QUBO → Ising**: Convert the QUBO matrix using `kw.core.qubo_matrix_to_ising_matrix()`
2. **Solve**: Run `Qmol_FM/sample.py` with the Ising matrix
3. **Ising → Binary**: Convert solutions back with `Qmol_FM/convert_ising_to_qubo.py`

**Note**: `user_id` and `sdk_code` are required credentials for the Kaiwu solver service. The Kaiwu SDK is optional and only needed for this pathway; GA and TPE generation do not require it.

## Citation

If you use this code in your research, please cite:

```
[Citation information to be added upon publication]
```

## License

[To be determined - add license information]

## Contact

For questions regarding this codebase, please contact:
- [Your contact information]

---

**Disclaimer**: This repository is for academic research and peer review purposes. Model weights and training data are not included.
