from __future__ import print_function
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import torch
import torch.optim
import argparse
from utils.inpainting_utils import *
from utils.denoising_utils import *
from models import *
from utils.quant import *
from utils.imp import *
import yaml

warnings.filterwarnings("ignore")

dtype = torch.cuda.FloatTensor
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def main(lr: float = 1e-2, max_steps: int=40000, sigma: float = 0.2, num_layers: int = 4, show_every: int = 1000, device_id: int = 0, 
         image_name: str = 'baboon', sparse: float = 0.5, prune_type: str = "rand_global"):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device()

    train_folder = 'images'
    img_np, img_noisy_np, noisy_psnr = load_image(train_folder, image_name, sigma)
    print(f"Noisy PSNR is '{noisy_psnr}'")
    print(f"Starting pruning at initialization ({prune_type}) training with sparsity {sparse} on image {image_name}")

    input_depth = 32
    output_depth = 3

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

    psnr, out = train_sparse(net, net_input, mask, img_np, img_noisy_np,
                             max_step=max_steps, show_every=show_every, device=device_id)

    outdir = f'images/{image_name}/pai/{prune_type}_{sparse}'
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
        'baboon', 'barbara', 'lena', 'pepper'
    ]

    parser.add_argument("--lr", type=float, help="the learning rate")
    parser.add_argument("--max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--sigma", type=float, help="noise-level")
    parser.add_argument("--num_layers", type=int, help="number of layers")
    parser.add_argument("--show_every", type=int, help="show every n steps")
    parser.add_argument("--device_id", type=int, help="specify which gpu")
    parser.add_argument("--image_name", type=str, choices=image_choices, help="name of image to denoise")
    parser.add_argument("--sparse", type=float, help="sparse percentage")
    parser.add_argument("--prune_type", type=str, help="pruning type")
    parser.add_argument("-f", "--file", type=str, default='configs/config_baseline_pai.yaml', help="YAML configuration file, options passed on the command line override these")

    args = parser.parse_args()

    default_config = {
        'lr': 1e-2,
        'max_steps': 40000,
        'sigma': 0.1,
        'num_layers': 6,
        'show_every': 1000,
        'device_id': 1,
        'image_name': 'baboon',
        'sparse': 0.5,
        'prune_type': 'rand_global'
    }

    config = set_config(args.file, default_config)

    main(
        lr=config.get('lr', default_config['lr']),
        max_steps=config.get('max_steps', default_config['max_steps']),
        sigma=config.get('sigma', default_config['sigma']),
        num_layers=config.get('num_layers', default_config['num_layers']),
        show_every=config.get('show_every', default_config['show_every']),
        device_id=config.get('device_id', default_config['device_id']),
        image_name=config.get('image_name', default_config['image_name']),
        sparse=config.get('sparse', default_config['sparse']),
        prune_type=config.get('prune_type', default_config['prune_type'])
    )

