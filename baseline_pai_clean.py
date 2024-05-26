from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
import sys
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from utils.denoising_utils import *
from models import *
from quant import *
from imp import *
from models.cnn import cnn
import torch
import torch.optim
import time
from utils.inpainting_utils import *
import pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sam import SAM

import argparse

def load_image(train_folder, image_name, sigma):
    train_noisy_folder = f'{train_folder}/train_noisy_{sigma}'
    os.makedirs(train_noisy_folder, exist_ok=True)
    file_path = os.path.join(train_folder, f'{image_name}.png')
    filename = os.path.splitext(os.path.basename(file_path))[0]
    img_pil = Image.open(file_path)
    img_pil = resize_and_crop(img_pil, max(img_pil.size))
    img_np = pil_to_np(img_pil)
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)
    img_noisy_pil.save(os.path.join(train_noisy_folder, filename + '.png'))
    noisy_psnr = compare_psnr(img_np, img_noisy_np)
    return img_np, img_noisy_np, noisy_psnr

def compare_psnr(img1, img2):
    mse = np.mean(np.abs(img1 - img2) ** 2)
    psnr = 10 * np.log10(np.max(np.abs(img1)) ** 2 / mse)
    return psnr

def main(lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2,
         num_layers: int = 4, show_every: int = 1000, device_id: int = 0, beta: float = 0.0,
         image_name: str = 'baboon', weight_decay: float = 0.0, sparse: float = 0.5, prune_type: str = "rand_global"):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device()

    train_folder = 'data/denoising/Set14'
    img_np, img_noisy_np, noisy_psnr = load_image(train_folder, image_name, sigma)
    print(f"Noisy PSNR is '{noisy_psnr}'")
    print(f"Starting pruning at initialization ({prune_type}) training with sparsity {sparse} on image {image_name}")

    input_depth = 32
    output_depth = 3

    mse = torch.nn.MSELoss().type(dtype)
    INPUT = "noise"

    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)
    net = skip(
        input_depth, output_depth,
        num_channels_down=[16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_up=[16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_skip=[0] * num_layers,
        upsample_mode='nearest',
        downsample_mode='avg',
        need1x1_up=False,
        filter_size_down=5,
        filter_size_up=3,
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad='reflection',
        act_fun='LeakyReLU'
    ).type(dtype)

    mask = get_pruning_mask(net)
    if prune_type == "rand_global":
        mask = prune_random_global(net, mask, sparse)
    elif prune_type == "rand_local":
        mask = prune_random_local(net, mask, sparse)
    elif prune_type == "mag_global":
        mask = prune_magnitude_global(net, mask, sparse)
    elif prune_type == "snip":
        mask = snip_prune(net, mask, net_input, img_np, img_noisy_np, sparse)
    elif prune_type == "grasp":
        mask = grasp_prune(net, mask, net_input, img_np, img_noisy_np, sparse)
    elif prune_type == "synflow":
        mask = synflow_prune(net, mask, net_input, sparse)
    elif prune_type == "snip_local":
        mask = snip_prune_local(net, mask, net_input, img_np, img_noisy_np, sparse)
    elif prune_type == "grasp_local":
        mask = grasp_prune_local(net, mask, net_input, img_np, img_noisy_np, sparse)
    elif prune_type == "synflow_local":
        mask = synflow_prune_local(net, mask, net_input, sparse)

    psnr, out = train_sparse(net, net_input, mask, img_np, img_noisy_np, max_step=max_steps, show_every=show_every, device=device_id)

    outdir = f'data/denoising/Set14/mask/{image_name}/pai/{prune_type}_{sparse}'
    print(f"Output directory: {outdir}")
    os.makedirs(f'{outdir}', exist_ok=True)

    with torch.no_grad():
        out_np = out
        img_var = np_to_torch(img_np)
        img_np = img_var.detach().cpu().numpy()
        psnr_gt = compare_psnr(img_np, out_np)
        print("PSNR of output image is: ", psnr_gt)
        np.savez(f'{outdir}/psnr_{image_name}.npz', psnr=psnr)

        output_paths = [
            f"{outdir}/out_{image_name}.png",
            f"{outdir}/img_np_{image_name}.png",
            f"{outdir}/img_noisy_np_{image_name}.png"
        ]

        print(out_np.shape, img_np.shape, img_noisy_np.shape)
        images_to_save = [out_np.transpose(1, 2, 0), img_np[0, :, :, :].transpose(1, 2, 0), img_noisy_np.transpose(1, 2, 0)]
        for path, img in zip(output_paths, images_to_save):
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        plt.plot(psnr)
        plt.title(f'PSNR vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('PSNR')
        plt.savefig(f'{outdir}/psnr_{image_name}_{sparse}_{prune_type}_{sigma}.png')
        plt.close()

    torch.cuda.empty_cache()
    print("Experiment done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")

    image_choices = [
        'baboon', 'barbara', 'bridge', 'coastguard', 'comic', 'face', 'flowers',
        'foreman', 'lenna', 'man', 'monarch', 'pepper', 'ppt3', 'zebra'
    ]

    parser.add_argument("--lr", type=float, default=1e-2, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=40000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--optim", type=str, default="ADAM", help="which optimizer")
    parser.add_argument("--reg", type=float, default=0.05, help="if regularization strength of igr")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise-level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show every n steps")
    parser.add_argument("--device_id", type=int, default=1, help="specify which gpu")
    parser.add_argument("--beta", type=float, default=0, help="momentum for sgd")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--image_name", type=str, choices=image_choices, default="baboon", help="name of image to denoise")
    parser.add_argument("--sparse", type=float, default=0.5, help="sparse percentage")
    parser.add_argument("--prune_type", type=str, default="rand_global", help="pruning type")

    args = parser.parse_args()

    main(lr=args.lr, max_steps=args.max_steps,
         optim=args.optim, reg=args.reg, sigma=args.sigma, num_layers=args.num_layers,
         show_every=args.show_every, beta=args.beta, device_id=args.device_id, image_name=args.image_name,
         weight_decay=args.decay, sparse=args.sparse, prune_type=args.prune_type)
