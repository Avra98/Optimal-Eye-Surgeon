# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import colormaps
# from natsort import natsorted

# def load_npz_file(file_path):
#     data = np.load(file_path)
#     # assuming your data is stored in arrays with specific keys, modify this line accordingly
#     y = data['psnr']
#     return y

# def plot_npz_files(ino):
#     files = []
#     plt.figure(figsize=(12, 6))
#     sparsities = [0.02, 0.03, 0.05, 0.5, 0.069]
#     cmap = colormaps['inferno']
#     for i, sparsity in enumerate(sparsities):
#         y = load_npz_file(f'data/denoising/face/mask/{ino}/unet/det/selected/{sparsity}/1e-09/out_sparsenet/0.1/psnr_{ino}.npz')
#         color = cmap(i / len(sparsities))
#         plt.plot(np.arange(len(y)), y, label=sparsity, color=color)

#     plt.xlabel('Iterations (thousands)')
#     plt.ylabel('PSNR')
#     plt.title(f'image {ino},noise=0.1')
#     plt.legend(loc='lower right')
#     plt.savefig(f'Kenneth_experiments/sparsenet_plots/sparsity_plot_{ino}.png')


# # replace 'your_directory_path' with the path to your target directory
# plt.rc('font', size=20)
# for i in [0, 2, 3, 4]:
#     plot_npz_files(i)
#     # plot_npz_files('data/denoising/Set14/mask/0/pat/14_0.2/')


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Corrected import

def load_npz_file(file_path):
    data = np.load(file_path)
    y = data['psnr']
    return y

def plot_npz_files(ino):
    files = []
    plt.figure(figsize=(12, 6))
    sparsities = [0.02, 0.03, 0.05, 0.5, 0.069]
    # sparsities = [0.0, 0.5, 0.8, 1.0, 10.0]
    cmap = cm.inferno  # Corrected usage
    for i, sparsity in enumerate(sparsities):
        y = load_npz_file(f'data/denoising/Dataset/mask/{ino}/sparsity/unet/det/{sparsity}/1e-09/out_sparsenet/0.1/psnr_{ino}.npz')
        color = cmap(i / len(sparsities))
        plt.plot(np.arange(len(y)), y, label=sparsity, color=color)

    plt.xlabel('Iterations (thousands)')
    plt.ylabel('PSNR')
    plt.title(f'Dataset_{ino}, noise=0.1')
    plt.ylim([min(y) - 1, max(y) + 1])  
    plt.yticks(np.arange(int(min(y)) - 1, int(max(y)) + 1, 1))  
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='lower right')
    plt.savefig(f'Kenneth_experiments/sparsenet_plots/sparsity_plot_Dataset{ino}.png')

plt.rc('font', size=20)
for i in [0, 1, 2, 3, 4]:
    plot_npz_files(i)

