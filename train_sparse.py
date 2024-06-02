from __future__ import print_function
import matplotlib.pyplot as plt
import os
import glob
import warnings
import numpy as np
from utils.denoising_utils import *
from models import *
from utils.quant import *
from utils.imp import *
from models.cnn import cnn
from PIL import Image
import torch
import torch.optim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sam import SAM
import argparse
import pickle as cPickle

# Suppress warnings
warnings.filterwarnings("ignore")

# Enable CUDA
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def main(image_name: str, lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2,
         num_layers: int = 4, show_every: int = 1000, device_id: int = 0, beta: float = 0.0,
         ino: int = 0, weight_decay: float = 0.0, mask_opt: str = "single", noise_steps: int = 80000, sparsity: float = 0.05):

    prior_sigma = inverse_sigmoid(sparsity)
    torch.cuda.set_device(device_id)
    torch.cuda.current_device()

    train_folder = 'images'
    img_np, img_noisy_np, noisy_psnr = load_image(train_folder, image_name, sigma)

    input_depth = 32
    output_depth = 3
    num_steps = noise_steps

    mse = torch.nn.MSELoss().type(dtype)
    net_input = get_noise(input_depth, "noise", img_np.shape[1:]).type(dtype)

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
        act_fun='LeakyReLU'
    ).type(dtype)

    outdir = f'sparse_models/{image_name}'
    os.makedirs(f'{outdir}/out_sparsenet/{sigma}', exist_ok=True)

    print(f"Starting sparse network training with mask with sparsity level '{sparsity}' on image '{image_name}' with sigma={sigma}.")
    print(f"All results will be saved in: {outdir}/out_sparsenet/{sigma}")

    with open(f'{outdir}/masked_model_{image_name}.pkl', 'rb') as f:
        masked_model = cPickle.load(f)
    with open(f'{outdir}/net_input_list_{image_name}.pkl', 'rb') as f:
        net_input_list = cPickle.load(f)
    with open(f'{outdir}/mask_{image_name}.pkl', 'rb') as f:
        mask = cPickle.load(f)

    masked_model = mask_network(mask, masked_model)

    psnr, out = train_sparse(masked_model, net_input_list, mask, img_np, img_noisy_np,
                             max_step=max_steps, show_every=show_every, device=device_id)

    with torch.no_grad():
        out_np = out
        img_var = np_to_torch(img_np)
        img_np = img_var.detach().cpu().numpy()
        psnr_gt = compare_psnr(img_np, out_np)
        print("PSNR of output image is: ", psnr_gt)
        np.savez(f'{outdir}/out_sparsenet/{sigma}/psnr_{ino}.npz', psnr=psnr)

        output_paths = [
            f"{outdir}/out_sparsenet/{sigma}/out_{image_name}.png",
            f"{outdir}/out_sparsenet/{sigma}/img_np_{image_name}.png",
            f"{outdir}/out_sparsenet/{sigma}/img_noisy_np_{image_name}.png"
        ]

        images_to_save = [out_np[0, :, :, :].transpose(1, 2, 0), img_np[0, :, :, :].transpose(1, 2, 0), img_noisy_np.transpose(1, 2, 0)]
        for path, img in zip(output_paths, images_to_save):
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()

        plt.plot(psnr)
        plt.title('PSNR vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('PSNR')
        plt.savefig(f'{outdir}/out_sparsenet/{sigma}/psnr_{ino}.png')
        plt.close()

    torch.cuda.empty_cache()
    print("Experiment done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using sparse DIP")

    image_choices = [
        'baboon', 'barbara', 'lenna', 'pepper'
    ]

    parser.add_argument("--image_name", type=str, choices=image_choices, default='pepper', help="which image to denoise")
    parser.add_argument("--lr", type=float, default=1e-2, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=40000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--optim", type=str, default="SAM", help="which optimizer")
    parser.add_argument("--reg", type=float, default=0.03, help="regularization strength")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show every N steps")
    parser.add_argument("--device_id", type=int, default=0, help="specify which GPU")
    parser.add_argument("--beta", type=float, default=0, help="momentum for SGD")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--mask_opt", type=str, default="det", help="mask type")

    args = parser.parse_args()

    main(image_name=args.image_name, lr=args.lr, max_steps=args.max_steps, optim=args.optim, reg=args.reg, sigma=args.sigma,
         num_layers=args.num_layers, show_every=args.show_every, beta=args.beta, device_id=args.device_id,  weight_decay=args.decay, mask_opt=args.mask_opt)