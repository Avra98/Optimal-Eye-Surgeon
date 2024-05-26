import os
from matplotlib.colors import Colormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from matplotlib import colormaps
import matplotlib.cm as cm 
from natsort import natsorted
# replace 'your_directory_path' with the path to your target directory

def load_npz_file(file_path):
    data = np.load(file_path)
    # assuming your data is stored in arrays with specific keys, modify this line accordingly
    y = data['psnr']
    return y

def plot_OES (ino):
    # Set14-0
    #npzfile = f'data/denoising/{dataset}/mask/{ino}/SAM(sigma=0.1,lr=0.01,decay=0,beta=0)/det/-1.8/1e-09/out_sparsenet/0.1/psnr_{ino}.npz' 
    # Set14-4
    npzfile = f'data/denoising/{dataset}/mask/{ino}/sparsity/det/0.05/1e-09/out_sparsenet/0.1/psnr_{ino}.npz'
    # Face-0
    #npzfile = f'data/denoising/{dataset}/mask/{ino}/unet/det/selected/0.02/1e-09/out_sparsenet/0.1/psnr_{ino}.npz'
    # Dataset-0
    # npzfile = f'data/denoising/{dataset}/mask/{ino}/sparsity/unet/det/0.03/1e-09/out_sparsenet/0.1/psnr_{ino}.npz'

    oes = load_npz_file(npzfile)
    print('OES fs ' , len(oes))
    plt.plot((epochs/len(oes))*np.arange(len(oes)), oes, label='OES (ours)', color='darkblue')

def plot_IMP (ino):
    # directory_imp = f'data/denoising/Dataset/mask/{ino}/pat/14_0.2/'
    directory_imp = f'data/denoising/Set14/mask/{ino}/pat/14_0.2/'
    files = natsorted([file for file in os.listdir(directory_imp) if file.endswith('.npz')])

    if not files:
        print("no .npz files found in the target directory.")
        return

    num_files = len(files)

    print('IMP fs ' , len(load_npz_file(os.path.join(directory_imp, files[0]))))
    for i, file in enumerate(files):
        # if (i in [0, 4, 6]): 
        #     continue
        file_path = os.path.join(directory_imp, file)
        y = load_npz_file(file_path)

        # calculate color based on position in the list
        color = cmap(i / (num_files - 1))

        plt.plot((epochs/len(y))*np.arange(len(y)), y, color=color)
    # plt.title(f'image {ino},noise=0.1')
    # plt.legend()

def plot_PAI (ino):
    # Set14 
    synflow = f'data/denoising/{dataset}/mask/{ino}/pai/synflow_local/sparse_0.9/psnr_{ino}.npz'
    grasp = f'data/denoising/{dataset}/mask/{ino}/pai/grasp_local/sparse_0.9/psnr_{ino}.npz'
    # Face
    # synflow = f'data/denoising/{dataset}/mask/{ino}/pai/synflow_local/psnr_{ino}.npz'
    # grasp = f'data/denoising/{dataset}/mask/{ino}/pai/grasp_local/psnr_{ino}.npz'
    # Dataset
    # synflow = f'data/denoising/{dataset}/mask/{ino}/pai/synflow_local_0.95/psnr_{ino}.npz'
    # grasp = f'data/denoising/{dataset}/mask/{ino}/pai/grasp_local_0.95/psnr_{ino}.npz'

    snpz = load_npz_file(synflow)
    gnpz = load_npz_file(grasp)
    print('Synflow fs ' , len(synflow))
    print('Grasp fs ' , len(grasp))
    plt.plot((epochs/(len(snpz)))*np.arange(len(snpz)), snpz, label='synflow')
    plt.plot((epochs/len(gnpz))*np.arange(len(gnpz)), gnpz, label='grasp')

dataset = 'Set14'
ino = 0
epochs = 4e4      
# cmap = mpl.cm.rainbow(np.linspace(0,1,20))
cmap = mpl.cm.inferno(np.linspace(0,0.8,20))
cmap = mpl.colors.ListedColormap(cmap[8:,:-1])
num_imp = 14

for ino in range(14):
    plt.rc('font', size=20)
    plt.figure(figsize=(15, 8))
    plt.xlabel('Number of Epochs')
    plt.ylabel('PSNR')
    plt.title('PSNR vs. Number of Epochs')
    plt.ylim(10,33)
    plt.grid()
    # plot_OES()
    plot_IMP(ino)
    # plot_PAI()
    # plt.legend(loc='upper right', bbox_to_anchor=(1.25,1))
    plt.legend(loc='lower right')

    norm = mpl.colors.Normalize(vmin=1, vmax=num_imp)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    plt.colorbar(sm, label='IMP iterations', ticks=np.arange(1,num_imp+1,2))
    # plt.subplots_adjust(right=0.8)
    plt.tight_layout()

    outdir = 'Kenneth_experiments/IMP_progression_Set14/'
    plt.savefig(f'{outdir}/Set14-imp_only-{ino}.png')
    plt.savefig(f'{outdir}/Set14-imp_only-{ino}.svg')
