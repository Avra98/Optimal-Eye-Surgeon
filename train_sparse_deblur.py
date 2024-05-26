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
from models.cnn import cnn
import torch
import torch.optim
import time
from PIL import Image
#from skimage.measure import compare_psnr
from utils.inpainting_utils import * 
import pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sam import SAM
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from scipy.ndimage import gaussian_filter


import argparse

def gaussian_kernel(size, sigma):
    """
    Generates a 2D Gaussian kernel.
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def convolve_with_gaussian(image_np, sigma):
    """
    Convolves a 3-channel image with a Gaussian kernel.
    Ensures the output image has the same size as the input.
    """
    if image_np.ndim == 3 and image_np.shape[0] == 3:  # Check if image has 3 channels
        blurred_image = np.zeros_like(image_np)
        kernel_size = int(sigma * 3) * 2 + 1  # Kernel size
        for c in range(3):
            blurred_image[c,...] = gaussian_filter(image_np[c,...], sigma=sigma, mode='reflect', truncate=3.0)
    else:
        kernel_size = int(sigma * 3) * 2 + 1
        blurred_image = gaussian_filter(image_np, sigma=sigma, mode='reflect', truncate=3.0)
    return blurred_image

def gaussian_kernel_torch(kernel_size, sigma):
    """
    Generates a 2D Gaussian kernel using PyTorch.
    """
    ax = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def convolve_with_gaussian_torch(image_tensor, sigma):
    """
    Convolves an image tensor with a Gaussian kernel using PyTorch.
    Assumes the image tensor is in BCHW format.
    """
    batch_size, channels, _, _ = image_tensor.shape
    kernel_size = int(sigma * 3) * 2 + 1
    kernel = gaussian_kernel_torch(kernel_size, sigma).to(image_tensor.device)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    
    # Ensure the kernel is [out_channels, in_channels/groups, height, width]
    padding = kernel_size // 2
    blurred_image = F.conv2d(image_tensor, kernel, padding=padding, groups=channels)
    
    return blurred_image


def add_salt_and_pepper_noise(image_np, amount=0.05, s_vs_p=0.5):
    """
    Add salt and pepper noise to a 3-channel image.
    Assumes image_np is in CHW format (3, height, width).
    """
    channels, rows, cols = image_np.shape
    noisy = np.copy(image_np)

    # Calculate the number of pixels to affect
    num_salt = np.ceil(amount * rows * cols * s_vs_p)
    num_pepper = np.ceil(amount * rows * cols * (1. - s_vs_p))

    # Apply salt noise (white pixels)
    for _ in range(int(num_salt)):
        x, y = np.random.randint(0, rows), np.random.randint(0, cols)
        noisy[:, x, y] = 1  # Set all channels at this pixel to white

    # Apply pepper noise (black pixels)
    for _ in range(int(num_pepper)):
        x, y = np.random.randint(0, rows), np.random.randint(0, cols)
        noisy[:, x, y] = 0  # Set all channels at this pixel to black

    return noisy


def main(images: list, lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2, num_layers: int = 4,
          show_every: int=1000, device_id: int = 0,beta: float = 0.0,ino : int =0,
          weight_decay: float = 0.0,sparsity: float = 0.1,kl: float = 0.1):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device() 
    amount=0.02
    sigma=1.0

    def normalize_image(img):
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val)   
    
    
    img_np_list=[]
    img_noisy_np_list=[]
    noisy_psnr_list=[]
    # train_folder = 'result/Urban100/image_SRF_2/train'
    train_folder = 'data/denoising/Set14'
    train_noisy_folder = 'data/deblurring/Set14/train_noisy_{}'.format(sigma)

    os.makedirs(train_noisy_folder, exist_ok=True)

    for i, file_path in enumerate(glob.glob(os.path.join(train_folder, '*.png'))):
        if i == ino:
            filename = os.path.splitext(os.path.basename(file_path))[0]
            imsize=-1
            img_pil = Image.open(file_path)
            img_pil = resize_and_crop(img_pil, max(img_pil.size))
            #img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_np = pil_to_np(img_pil)
            print("img_np is:",img_np.shape)                 
            # # Apply Gaussian blur
            # img_blurred_np = convolve_with_gaussian(img_np, sigma=sigma)
            # print("img_blurred is:",img_blurred_np.shape)
            # img_blurred_pil = np_to_pil(img_blurred_np)
            # img_blurred_pil.save(os.path.join(train_noisy_folder, f"{filename}_blurred.png"))
            
            # Add salt and pepper noise
            img_noisy_np = add_salt_and_pepper_noise(img_np, amount=amount, s_vs_p=0.5)
            print("img_noisy is:",img_noisy_np.shape)
            img_noisy_pil = np_to_pil(img_noisy_np)
            img_noisy_pil.save(os.path.join(train_noisy_folder, f"{filename}_noisy.png"))

            break
         
    #print(img_np.shape,img_noisy_np.shape,img_blurred_np.shape)        
    noisy_psnr = compare_psnr(img_np,img_noisy_np)
    print("noisy psnr:",noisy_psnr)
    noisy_psnr_list.append(noisy_psnr)
    print(f'Starting vanilla DIP on {ino} using {optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta})')
    print(f"Noisy PSNR is '{noisy_psnr}'")            

    # Modify input and output depths
    input_depth = 32   
    output_depth = 3

    # Adjust loss function
    mse = torch.nn.MSELoss().type(dtype)
    INPUT = "noise"        
    net_input= get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype) 
    print("input dim:", net_input.shape) #[1, 3, 256, 256]
    # net_input = 
    net = skip(
        input_depth, output_depth,
        num_channels_down = [16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_up   = [16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_skip = [0]*num_layers,
        upsample_mode='nearest',
        downsample_mode='avg',
        need1x1_up = False,
        filter_size_down=5, 
        filter_size_up=3,
        filter_skip_size = 1,
        need_sigmoid=True, 
        need_bias=True, 
        pad='reflection', 
        act_fun='LeakyReLU').type(dtype)

    outdir = f'data/deblurring/Set14/mask/{ino}/sparsity/det/{sparsity}/{kl}/6layers'
    print(f"Output directory where image taken from: {outdir}")
    os.makedirs(f'{outdir}/out_sparsenet/just_salt', exist_ok=True)
    print(f"Output directory where results stored: {outdir}/out_sparsenet/just_salt")
    #     


    # load masked model and net_input_list from the outdir folder
    with open(f'{outdir}/masked_model_{ino}.pkl', 'rb') as f:
         masked_model = cPickle.load(f)
    with open(f'{outdir}/net_input_list_{ino}.pkl', 'rb') as f:
        net_input_list = cPickle.load(f)
    # load the saved mask
    with open(f'{outdir}/mask_{ino}.pkl', 'rb') as f:
        mask = cPickle.load(f)
    net_input = net_input_list.to(device_id)
    masked_model = masked_model.to(device_id)
    mask = mask.to(device_id)
    optimizer = torch.optim.Adam(masked_model.parameters(), lr=lr, weight_decay = weight_decay)
    i=0
   #[1e-1,1e-2,5e-2],[0.5,0.8] [1e-3 5e-3]
    tot_loss = []
    grad_list = []
    sharp=[]
    psnr_list=[]
    img_var = np_to_torch(img_np).to(device_id)
    noise_var = np_to_torch(img_noisy_np).to(device_id)
    for epoch in range(max_steps):
                # start_time = time.time()
        optimizer.zero_grad()
        out = masked_model(net_input)
        #blurred_out = convolve_with_gaussian_torch(out, sigma)
        # with torch.no_grad():  
        #     print(f"device of blurred_out is {blurred_out.device} (index {blurred_out.device.index})")
        #     if next(masked_model.parameters(), None) is not None:
        #         print(f"device of masked_model is {next(masked_model.parameters()).device} (index {next(masked_model.parameters()).device.index})")
        #     else:
        #         print("masked_model has no parameters or is not on CUDA.")
        #     print(f"device of mask is {mask.device} (index {mask.device.index})")
        #     print(f"device of noise_var is {noise_var.device} (index {noise_var.device.index})")
        #     print(f"device of out is {out.device} (index {out.device.index})")


        total_loss = mse(out, noise_var)      
        total_loss.backward()
        k = 0
        for param in masked_model.parameters():
            t = len(param.view(-1))
            param.grad.data = param.grad.data * mask[k:(k+t)].view(param.grad.data.shape)
            k += t

        optimizer.step()
        with torch.no_grad():    

            if epoch%show_every==0 and epoch!=0:
                out_np = out.detach().cpu().numpy()
                img_np = img_var.detach().cpu().numpy()
                psnr  = compare_psnr(img_np, out_np)
                print(f"At step '{epoch}', psnr is '{psnr}'")
                psnr_list.append(psnr)  


    plt.plot(psnr_list)
    plt.savefig(f'{outdir}/out_sparsenet/just_salt/psnr_{ino}.png')
    plt.close()
    ## save the list "psnr" as an npz file and save in the outdir folder
    np.savez(f'{outdir}/out_sparsenet/just_salt/psnr_{ino}.npz', psnr=psnr_list)


    ## save the final output image as a png file and save in the outdir folder
    out_np = out.detach().cpu().numpy()
    #out_np = out_np.transpose(1, 2, 0)
    plt.imshow(out_np)
    plt.imsave(f'{outdir}/out_sparsenet/just_salt/out_{ino}.png', out_np)
    plt.close()

    

    torch.cuda.empty_cache()
    print("Experiment done")           
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")
    
    parser.add_argument("--images", type=str, default = ["Lena512rgb"], help="which image to denoise")
    parser.add_argument("--lr", type=float,  default=1e-4, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=40000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--optim", type=str, default="ADAM", help="which optimizer")
    #parser.add_argument("--IGR", type=str, default="Normal", help="true if SAM ")
    parser.add_argument("--reg", type=float, default=0.05, help="if regularization strength of igr")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise-level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show_every")
    parser.add_argument("--device_id", type=int, default=1, help="specify which gpu")
    parser.add_argument("--beta", type=float, default=0, help="momentum for sgd ")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--ino", type=int, default=0, help="image index ")
    parser.add_argument("--sparsity", type=float, default=0.05, help="sparsity")
    parser.add_argument("--kl", type=float, default=1e-9, help="kl") 
    args = parser.parse_args()
    
    main(images=args.images, lr=args.lr, max_steps=args.max_steps, optim= args.optim,reg=args.reg,sigma = args.sigma, 
         num_layers = args.num_layers, show_every = args.show_every, beta = args.beta, 
         device_id = args.device_id,ino = args.ino, weight_decay = args.decay, sparsity = args.sparsity, kl = args.kl)
        
    
    
        
        
    
    

