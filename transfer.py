from __future__ import print_function
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import torch
import torch.optim
import pickle as cPickle
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.inpainting_utils import * 
from utils.denoising_utils import *
from utils.quant import *
from utils.imp import *
from models import *
import yaml

warnings.filterwarnings("ignore")

dtype = torch.cuda.FloatTensor
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True

def main(sigma: float = 0.1, num_layers: int = 4, show_every: int=1000, device_id: int = 0, 
          image_name: str = "baboon", trans_type: str="pai", transferimage_name: str = "barbara",
          max_steps: int = 80000, sparsity: float = 0.05):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device()

    train_folder = 'images'
    img_np, img_noisy_np, noisy_psnr = load_image(train_folder, image_name, sigma)
    print(f"Noisy PSNR is '{noisy_psnr}'")

    print(f"Performing mask transfer operation for {image_name} using {transferimage_name}'s mask with sparsity {sparsity}")
    input_depth = 32
    output_depth = 3

    INPUT = "noise"

    net_input_list = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)
    masked_model = skip(
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
        act_fun='LeakyReLU').type(dtype)

    if trans_type == "pai":
        outdir = f'sparse_models/{transferimage_name}'
        print(f"Output directory: {outdir}")
        os.makedirs(f'{outdir}/trans_{transferimage_name}_sparsenet_set14/{sigma}', exist_ok=True)
        with open(f'{outdir}/net_input_list_{transferimage_name}.pkl', 'rb') as f:
            net_input_list = cPickle.load(f)
        with open(f'{outdir}/mask_{transferimage_name}.pkl', 'rb') as f:
            mask = cPickle.load(f)

    elif trans_type == "pat":
        outdir = f'sparse_models_imp/{transferimage_name}'
        print(f"Output directory: {outdir}")
        os.makedirs(f'{outdir}/trans_{transferimage_name}_sparsenet_set14/{sigma}', exist_ok=True)
        model_path = f'{outdir}/model_final.pth'
        net_input_list = torch.load(f'{outdir}/net_input_final.pth')
        mask = torch.load(f'{outdir}/mask_final.pth')

    masked_model = mask_network(mask, masked_model)

    psnr, out = train_sparse(masked_model, net_input_list, mask, img_np, img_noisy_np, max_step=max_steps, show_every=show_every, device=device_id)

    with torch.no_grad():
        out_np = out
        img_var = np_to_torch(img_np)
        img_np = img_var.detach().cpu().numpy()
        psnr_gt = compare_psnr(img_np, out_np)
        print("PSNR of output image is: ", psnr_gt)
        np.savez(f'{outdir}/trans_{transferimage_name}_sparsenet_set14/{sigma}/psnr_{image_name}.npz', psnr=psnr)

        output_paths = [
            f"{outdir}/trans_{image_name}_sparsenet_set14/{sigma}/out_{image_name}.png",
            f"{outdir}/trans_{image_name}_sparsenet_set14/{sigma}/img_np_{image_name}.png",
            f"{outdir}/trans_{image_name}_sparsenet_set14/{sigma}/img_noisy_np_{image_name}.png"]

        print(out_np.shape, img_np.shape, img_noisy_np.shape)
        images_to_save = [out_np[0, :, :, :].transpose(1, 2, 0), img_np[0, :, :, :].transpose(1, 2, 0), img_noisy_np.transpose(1, 2, 0)]
        for path, img in zip(output_paths, images_to_save):
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()
            plt.plot(psnr)
            plt.title(f'PSNR vs Iterations')
            plt.xlabel('Iterations')
            plt.ylabel('PSNR')
            plt.savefig(f'{outdir}/trans_{transferimage_name}_sparsenet_set14/{sigma}/psnr_{image_name}.png')
            plt.close()

    torch.cuda.empty_cache()
    print("Transfer done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")

    image_choices = [
        'baboon', 'barbara', 'lena', 'pepper'
    ]

    parser.add_argument("--max_steps", type=int, default=40000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise-level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show_every")
    parser.add_argument("--device_id", type=int, default=0, help="specify which gpu")
    parser.add_argument("--image_name", type=str, choices=image_choices, default="baboon", help="image to denoise")
    parser.add_argument("--transferimage_name", type=str, choices=image_choices, default="barbara", help="transfer image from which to transfer")
    parser.add_argument("--trans_type", type=str, default="pai", help="transfer type")
    parser.add_argument("--sparsity", type=float, default=0.05, help="sparsity percent")
    parser.add_argument("-f", "--file", type=str, default='configs/config_transfer.yaml', help="YAML configuration file, options passed on the command line override these")
    args = parser.parse_args()

    default_config = {
        'max_steps': 40000,
        'sigma': 0.1,
        'num_layers': 6,
        'show_every': 1000,
        'device_id': 0,
        'image_name': 'baboon',
        'transferimage_name': 'barbara',
        'trans_type': 'pai',
        'sparsity': 0.05
    }

    config = {}
    if args.file:
        try:
            with open(args.file, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f'Config file {args.file} not found. Using default values.')
            # Write the default config to the specified config file
            with open(args.file, 'w') as file:
                yaml.dump(default_config, file)
            print(f"Default configuration file '{args.file}' has been created.")

    # Override config with command line arguments if provided
    config.update({k: v for k, v in vars(args).items() if v is not None})

    main(
        max_steps=config.get('max_steps', default_config['max_steps']),
        sigma=config.get('sigma', default_config['sigma']),
        num_layers=config.get('num_layers', default_config['num_layers']),
        show_every=config.get('show_every', default_config['show_every']),
        device_id=config.get('device_id', default_config['device_id']),
        image_name=config.get('image_name', default_config['image_name']),
        transferimage_name=config.get('transferimage_name', default_config['transferimage_name']),
        trans_type=config.get('trans_type', default_config['trans_type']),
        sparsity=config.get('sparsity', default_config['sparsity'])
    )



