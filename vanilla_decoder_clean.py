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
#from torch.autograd import Variable
from utils.denoising_utils import *
from models import *
from models.decoder import decodernw,resdecoder
import torch
import torch.optim
import time
from PIL import Image
from torch.autograd import Variable
#from skimage.measure import compare_psnr
from utils.inpainting_utils import * 
from utils.quant import *
from utils.imp import *
import pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sam import SAM

import argparse
# Suppress warnings
warnings.filterwarnings("ignore")

# Enable CUDA
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor



def main(image_name: str, lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2,
         num_layers: int = 4, show_every: int = 1000, device_id: int = 0, beta: float = 0.0,
         k: int = 5, weight_decay: float = 0.0):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device()

    train_folder = 'data/denoising/Set14'
    img_np, img_noisy_np, noisy_psnr = load_image(train_folder, image_name, sigma)

    print(f"Starting reconstruction by deep decoder with {k} layers on image '{image_name}")
    print(f"Noisy PSNR is '{noisy_psnr}'")

    output_depth = 3
    num_channels = [128] * k
    psrn_noisy_last = 0

    mse = torch.nn.MSELoss().type(dtype)
    totalupsample = 2**len(num_channels)
    width = int(img_np.shape[1] / totalupsample)
    height = int(img_np.shape[1] / totalupsample)
    shape = [1, num_channels[0], width, height]

    net_input = torch.zeros(shape).uniform_().type(dtype)
    net = decodernw(output_depth, num_channels_up=num_channels, upsample_first=True).type(dtype)

    s = sum(np.prod(list(p.size())) for p in net.parameters())
    print('Number of params in decoder: %d' % s)

    print(f"Starting optimization with optimizer '{optim}'")
    if optim == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay, momentum=beta)
    elif optim == "ADAM":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    elif optim == "SAM":
        base_opt = torch.optim.SGD
        optimizer = SAM(net.parameters(), base_opt, rho=reg, adaptive=False, lr=lr, weight_decay=weight_decay, momentum=beta)

    psnr_list = []

    def closure_sgd(net_input, img_var, noise_var):
        img_var = np_to_torch(img_var).type(dtype)
        noise_var = np_to_torch(noise_var).type(dtype)
        out = net(net_input)
        total_loss = mse(out, noise_var)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        out_np = out.detach().cpu().numpy()
        img_np = img_var.detach().cpu().numpy()
        psnr_gt = compare_psnr(img_np, out_np)
        return psnr_gt, out_np

    
    outdir = f'data/denoising/Dataset/mask/{image_name}/deepdecoder_{k}/{sigma}'
    os.makedirs(f'{outdir}', exist_ok=True)

    for j in range(max_steps):
        psnr, out = closure_sgd(net_input, img_np, img_noisy_np)
        psnr_noisy = compare_psnr(img_noisy_np, out[0, :, :, :])

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

    print(out.shape, img_np.shape, img_noisy_np.shape)
    images_to_save = [out[0, :, :, :].transpose(1, 2, 0), img_np.transpose(1, 2, 0), img_noisy_np.transpose(1, 2, 0)]
    for path, img in zip(output_paths, images_to_save):
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

    plt.plot(psnr)
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
        'baboon', 'barbara', 'bridge', 'coastguard', 'comic', 'face', 'flowers',
        'foreman', 'lenna', 'man', 'monarch', 'pepper', 'ppt3', 'zebra'
    ]

    parser.add_argument("--image_name", type=str, choices=image_choices, default='pepper', required=False, help="which image to denoise")
    parser.add_argument("--lr", type=float, default=1e-2, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=40000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--optim", type=str, default="ADAM", help="which optimizer")
    parser.add_argument("--reg", type=float, default=0.05, help="regularization strength")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show every N steps")
    parser.add_argument("--device_id", type=int, default=1, help="specify which GPU")
    parser.add_argument("--beta", type=float, default=0, help="momentum for SGD")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--k", type=int, default=5, help="number of channels")

    args = parser.parse_args()

    main(image_name=args.image_name, lr=args.lr, max_steps=args.max_steps, optim=args.optim, reg=args.reg, sigma=args.sigma,
         num_layers=args.num_layers, show_every=args.show_every, beta=args.beta, device_id=args.device_id,
         k=args.k, weight_decay=args.decay)
