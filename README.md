## Optimal Eye Surgeon 

This repository contains the source code for pruning image generator networks at initialization. 

## Table of Contents
- [One-time Setup](#one-time-setup)
  - [Installation](#installation)
  - [Environment Setup](#environment-setup)
- [Updating](#updating)

- [Working with the Code](#working-with-the-code)
  - [Finding-1: Finding mask at initialization](#finding-1-finding-mask-at-initialization)
  - [Finding-2: Sparse network training](#finding-2-sparse-network-training)
  - [Finding-3: Sparse network transfer](#finding-3-sparse-network-transfer)
    - [Transfer OES masks](#transfer-oes-masks)
    - [Transfer IMP masks](#transfer-imp-masks)
  - [Finding-4: Baseline pruning methods](#finding-4-baseline-pruning-methods)
    - [Pruning at initialization Methods](#pruning-at-initialization-methods)
    - [IMP](#imp)


- [Contributing](#contributing)
- [License](#license)


## One-time Setup
1. Install a working version of conda.
2. Create a conda environment: `conda env create --file environment.yml`
3. Activate the conda environment: `conda activate lot`

## Updating
1. Pull the latest changes: `git pull`
2. Update the conda environment: `conda env update -f environment.yml`

## Working with the Code
Activate the conda environment before working with the code: `conda activate lot`

### Finding-1: Finding mask at initialization
To find the mask at initialization, follow these steps:
```python
python find_mask.py --config config_file.yaml


### Finding-2: Sparse network training
To train the sparse network, follow these steps:
1. Navigate to the training directory:
   \```bash
   cd path/to/training/scripts
   \```
2. Run the training script with the appropriate configuration:
   \```bash
   python train_sparse_network.py --config config_file.yaml
   \```

### Finding-3: Sparse network transfer
####  Transfer OES masks
To transfer OES masks to another network:
1. Navigate to the OES transfer directory:
   \```bash
   cd path/to/oes/transfer/scripts
   \```
2. Run the transfer script:
   \```bash
   python transfer_oes_masks.py --source_config source_config.yaml --target_config target_config.yaml
   \```

####  Transfer IMP masks
To transfer IMP masks to another network:
1. Navigate to the IMP transfer directory:
   \```bash
   cd path/to/imp/transfer/scripts
   \```
2. Run the transfer script:
   \```bash
   python transfer_imp_masks.py --source_config source_config.yaml --target_config target_config.yaml
   \```

### Finding-4: Baseline pruning methods
#### Pruning at initialization Methods
To perform pruning at initialization:
1. Navigate to the pruning at initialization directory:
   \```bash
   cd path/to/pruning/initialization/scripts
   \```
2. Run the pruning script:
   \```bash
   python prune_at_initialization.py --config config_file.yaml
   \```

#### IMP
To perform Iterative Magnitude Pruning (IMP):
1. Navigate to the IMP directory:
   \```bash
   cd path/to/imp/scripts
   \```
2. Run the IMP script:
   \```bash
   python iterative_magnitude_pruning.py --config config_file.yaml
   \```
