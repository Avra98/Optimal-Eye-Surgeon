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
    - [Branch-1: Transfer OES masks](#branch-1-transfer-oes-masks)
    - [Branch-2: Transfer IMP masks](#branch-2-transfer-imp-masks)
  - [Finding-4: Baseline pruning methods](#finding-4-baseline-pruning-methods)
    - [Branch-1: Pruning at initialization Methods](#branch-1-pruning-at-initialization-methods)
    - [Branch-2: IMP](#branch-2-imp)


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

