import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from natsort import natsorted

def load_npz_file(file_path):
    data = np.load(file_path)
    # assuming your data is stored in arrays with specific keys, modify this line accordingly
    y = data['psnr']
    return y

def plot_npz_files(ino):
    # directory = f'data/denoising/Set14/mask/{ino}/pat/14_0.2/'
    directory_imp = f'data/denoising/Dataset/mask/{ino}/pat/14_0.2/'
    directory_oes =
    files = natsorted([file for file in os.listdir(directory_imp) if file.endswith('.npz')])

    if not files:
        print("no .npz files found in the target directory.")
        return

    cmap = colormaps['inferno']
    num_files = len(files)

    plt.figure(figsize=(11, 6))

    for i, file in enumerate(files):
        file_path = os.path.join(directory_imp, file)
        y = load_npz_file(file_path)

        # calculate color based on position in the list
        color = cmap(i / num_files)

        plt.plot(1000*np.arange(len(y)), y, label=file, color=color)

    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.title(f'image {ino},noise=0.1')
    # plt.legend()
    plt.savefig(f'Kenneth_experiments/IMP_progression_Dataset/IMP_progression_{ino}.png')


# replace 'your_directory_path' with the path to your target directory
plt.rc('font', size=20)
for i in range(6):
# for i in [0, 2, 3, 4]:
    plot_npz_files(i)
    # plot_npz_files('data/denoising/Set14/mask/0/pat/14_0.2/')
