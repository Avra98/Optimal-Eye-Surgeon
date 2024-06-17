from __future__ import print_function
from sam import SAM
import argparse
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import pickle as cPickle
from utils.inpainting_utils import *
from PIL import Image
import time
import torch.optim
import torch
from models.cnn import cnn
from models import *
from utils.imp import *
from utils.quant import *
from utils.sharpness import *
from utils.denoising_utils import *
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


dtype = torch.cuda.FloatTensor


def main(lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.1, num_layers: int = 6,
         show_every: int = 1000, device_id: int = 0, beta: float = 0.0, image_name: str = 'pepper',
         weight_decay: float = 0.0):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device()

    train_folder = 'images'
    img_np, img_noisy_np, noisy_psnr = load_image(
        train_folder, image_name, sigma)
    print("noisy psnr:", noisy_psnr)
    print(f'Starting vanilla DIP on {image_name} using {
          optim}(sigma={sigma}, lr={lr}, decay={weight_decay}, beta={beta})')
    print(f"Noisy PSNR is '{noisy_psnr}'")

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

    if optim == "SGD":
        optimizer = torch.optim.SGD(
            net.parameters(), lr=lr, weight_decay=weight_decay, momentum=beta)
    elif optim == "ADAM":
        optimizer = torch.optim.Adam(
            net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim == "SAM":
        base_opt = torch.optim.SGD
        optimizer = SAM(net.parameters(), base_opt, rho=reg, adaptive=False,
                        lr=lr, weight_decay=weight_decay, momentum=beta)

    def closure_sgd(net_input, img_var, noise_var):
        optimizer.zero_grad()
        img_var = np_to_torch(img_var).type(dtype)
        noise_var = np_to_torch(noise_var).type(dtype)
        out = net(net_input)
        total_loss = mse(out, noise_var)
        total_loss.backward()
        optimizer.step()
        out_np = out.detach().cpu().numpy()
        img_np = img_var.detach().cpu().numpy()
        psnr_gt = compare_psnr(img_np, out_np)
        return psnr_gt, out_np

    outdir = f'images/{image_name}/vanilla/{sigma}'
    os.makedirs(f'{outdir}', exist_ok=True)

    psnr_list = []
    for j in range(max_steps):
        psnr, out = closure_sgd(net_input, img_np, img_noisy_np)
        if j % show_every == 0 and j != 0:
            print(f"At step '{j}', psnr is '{psnr}'")
            psnr_list.append(psnr)

    plt.plot(psnr_list)
    plt.savefig(f'{outdir}/psnr_{image_name}.png')
    plt.close()
    np.savez(f'{outdir}/psnr_{image_name}.npz', psnr=psnr_list)

    torch.cuda.empty_cache()
    print("Experiment done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")

    image_choices = [
        'pepper', 'lena', 'barbara', 'baboon'
    ]

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=40000,
                        help="the maximum number of gradient steps to train for")
    parser.add_argument("--optim", type=str, default="ADAM",
                        help="which optimizer")
    parser.add_argument("--reg", type=float, default=0.05,
                        help="if regularization strength of igr")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise-level")
    parser.add_argument("--num_layers", type=int,
                        default=6, help="number of layers")
    parser.add_argument("--show_every", type=int,
                        default=100, help="show every n steps")
    parser.add_argument("--beta", type=float, default=0,
                        help="momentum for sgd")
    parser.add_argument("--device_id", type=int, default=1,
                        help="specify which gpu")
    parser.add_argument("--image_name", type=str, choices=image_choices,
                        default="pepper", help="name of image to denoise")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")

    args = parser.parse_args()

    main(lr=args.lr, max_steps=args.max_steps, optim=args.optim, reg=args.reg, sigma=args.sigma,
         num_layers=args.num_layers, show_every=args.show_every, beta=args.beta, device_id=args.device_id,
         image_name=args.image_name, weight_decay=args.decay)
