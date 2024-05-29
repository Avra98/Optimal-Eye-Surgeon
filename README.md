## Optimal Eye Surgeon (ICML-2024)

<div style="display: flex; justify-content: space-around;">
    <img src="paper_figures/flow.svg" alt="Flow Diagram" style="width: 45%;"/>
    <img src="paper_figures/transfer.svg" alt="Transfer Diagram" style="width: 45%;"/>
</div>

This repository contains the source code for pruning image generator networks at initialization to alleviate overfitting.


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


![equation](paper_figures/equation.png){width=50px}

The following code implements the above optimization using Gumbel softmax reparameterization trick:

```python
python dip_mask_clean.py --image_name="pepper" --sparsity=0.05
```

to generate supermasks at various sparsity levels as follows

![supermasks](paper_figures/only2masks.svg){width=50px}




### Finding-2: Sparse network training

Implements sparse network training on the image.

```python
python train_sparse_clean.py --image_name="pepper"
```


### Finding-3: Sparse network transfer
####  Transfer OES masks

![Sparse Network Transfer 1](paper_figures/another.gif)
![Sparse Network Transfer 2](paper_figures/Lena_ppt3.gif)



### Finding-4: Baseline pruning methods
#### Pruning at initialization Methods
```python
python find_mask.py --config config_file.yaml
```

#### IMP
```python
python find_mask.py --config config_file.yaml
```