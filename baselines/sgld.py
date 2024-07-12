from __future__ import print_function
import matplotlib.pyplot as plt
import os
import warnings
import numpy as np
import torch
import torch.optim
from utils.inpainting_utils import *
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import argparse
from utils.denoising_utils import *
from models import *
from utils.quant import *
from utils.imp import *

# Suppress warnings
warnings.filterwarnings("ignore")

# Enable CUDA
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def add_noise(model, param_noise_sigma, lr):
    for n in [x for x in model.parameters() if len(x.size()) == 4]:
        noise = torch.randn(n.size()).to(n.device) * param_noise_sigma * lr
        n.data = n.data + noise


def main(image_name: str, lr: float, max_steps: int, reg: float = 0.0, sigma: float = 0.2,
         num_layers: int = 4, show_every: int = 1000, device_id: int = 0, beta: float = 0.0,
         weight_decay: float = 0.0):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device()

    train_folder = 'images'
    img_np, img_noisy_np, noisy_psnr = load_image(
        train_folder, image_name, sigma)

    print(f"Starting SGLD with image {image_name} ")
    print(f"Noisy PSNR is '{noisy_psnr}'")

    input_depth = 32
    output_depth = 3
    param_noise_sigma = 0.5

    mse = torch.nn.MSELoss().type(dtype)
    net_input = get_noise(input_depth, "noise", img_np.shape[1:]).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

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

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    psnr_list = []
    reg_noise_std = 1. / 30.

    def closure_sgld(net_input, img_var, noise_var):
        nonlocal net_input_saved, noise, reg_noise_std
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
        img_var = np_to_torch(img_var).type(dtype)
        noise_var = np_to_torch(noise_var).type(dtype)
        out = net(net_input)
        total_loss = mse(out, noise_var)
        total_loss.backward()
        out_np = out.detach().cpu().numpy()[0]
        psnr_gt = compare_psnr(img_np, out_np)
        return psnr_gt, out_np

    outdir = f'images/{image_name}/sgld'
    os.makedirs(f'{outdir}', exist_ok=True)

    for j in range(max_steps):
        optimizer.zero_grad()
        psnr, out = closure_sgld(net_input, img_np, img_noisy_np)
        psnr_noisy = compare_psnr(img_noisy_np, out)
        optimizer.step()
        add_noise(net, param_noise_sigma, lr)

        if j % show_every == 0 and j != 0:
            print(f"At step '{j}', psnr is '{psnr}', noisy psnr is '{psnr_noisy}'")
            psnr_list.append(psnr)

    plt.plot(psnr_list)
    plt.savefig(f'{outdir}/psnr_{image_name}.png')
    plt.close()
    np.savez(f'{outdir}/psnr_{image_name}.npz', psnr=psnr_list)

    output_paths = [
        f"{outdir}/out_{image_name}.png",
        f"{outdir}/img_np_{image_name}.png",
        f"{outdir}/img_noisy_np_{image_name}.png"
    ]

    images_to_save = [out.transpose(1, 2, 0), img_np.transpose(
        1, 2, 0), img_noisy_np.transpose(1, 2, 0)]
    for path, img in zip(output_paths, images_to_save):
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

    plt.plot(psnr_list)
    plt.title('PSNR vs Iterations')
    plt.xlabel('Iterations')
    plt.ylabel('PSNR')
    plt.savefig(f'{outdir}/psnr_{image_name}.png')
    plt.close()

    torch.cuda.empty_cache()
    print("Experiment done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")

    image_choices = [
        'baboon', 'barbara', 'lena', 'pepper'
    ]

    parser.add_argument("--image_name", type=str, choices=image_choices,
                        default='pepper', required=False, help="which image to denoise")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=40000,
                        help="the maximum number of gradient steps to train for")
    parser.add_argument("--reg", type=float, default=0.05,
                        help="regularization strength")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise level")
    parser.add_argument("--num_layers", type=int,
                        default=6, help="number of layers")
    parser.add_argument("--show_every", type=int,
                        default=1000, help="show every N steps")
    parser.add_argument("--device_id", type=int, default=1,
                        help="specify which GPU")
    parser.add_argument("--beta", type=float, default=0,
                        help="momentum for SGD")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")

    args = parser.parse_args()

    main(image_name=args.image_name, lr=args.lr, max_steps=args.max_steps, reg=args.reg, sigma=args.sigma,
         num_layers=args.num_layers, show_every=args.show_every, beta=args.beta, device_id=args.device_id,
         weight_decay=args.decay)
