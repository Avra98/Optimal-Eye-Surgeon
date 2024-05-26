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
from models import *
from models.decoder import decodernw,resdecoder
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


# def pil_to_np(image):
#     """
#     Convert a PIL Image to a NumPy array.
#     """
#     return np.asarray(image) / 255.0

# def np_to_pil(image_np):
#     """
#     Convert a NumPy array to a PIL Image.
#     """
#     return Image.fromarray((image_np * 255).astype(np.uint8))

def main(images: list, lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2, num_layers: int = 4, show_every: int=1000, device_id: int = 0,beta: float = 0.0,ino : int =0,weight_decay: float = 0.0):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device() 
    amount=0.07
    sigma=2.0

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
#    for image in images:
#        imagename = "image_" + str(image) + ".png"
#        fname = 'data/denoising/Dataset' + "/" + imagename
#        imsize = -1
#        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
#        img_np = pil_to_np(img_pil)
#        img_np = img_np[0, :, :]
#        img_noisy_np = img_np + sigma*np.random.normal(scale=sigma, size=img_np.shape)
#        img_noisy_np = normalize_image(img_noisy_np)                
#        img_np_list.append(img_np)
#        img_noisy_np_list.append(img_noisy_np)  

    for i, file_path in enumerate(glob.glob(os.path.join(train_folder, '*.png'))):
        if i == ino:
            filename = os.path.splitext(os.path.basename(file_path))[0]
            imsize=-1
            img_pil = Image.open(file_path)
            img_pil = resize_and_crop(img_pil, max(img_pil.size))
            #img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_np = pil_to_np(img_pil)
            print("img_np is:",img_np.shape)
                        
            # Apply Gaussian blur
            img_blurred_np = convolve_with_gaussian(img_np, sigma=sigma)
            print("img_blurred is:",img_blurred_np.shape)
            img_blurred_pil = np_to_pil(img_blurred_np)
            img_blurred_pil.save(os.path.join(train_noisy_folder, f"{filename}_blurred.png"))
            
            # Add salt and pepper noise
            img_noisy_np = add_salt_and_pepper_noise(img_blurred_np, amount=amount, s_vs_p=0.5)
            print("img_noisy is:",img_noisy_np.shape)
            # Convert the noisy image bac
            # k to PIL for saving

            img_noisy_pil = np_to_pil(img_noisy_np)
            img_noisy_pil.save(os.path.join(train_noisy_folder, f"{filename}_noisy.png"))

            break

            
    print(img_np.shape,img_noisy_np.shape,img_blurred_np.shape)        
    noisy_psnr = compare_psnr(img_np,img_noisy_np)
    print("noisy psnr:",noisy_psnr)
    noisy_psnr_list.append(noisy_psnr)
    print(f'Starting vanilla DIP on {ino} using {optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta})')
    print(f"Noisy PSNR is '{noisy_psnr}'")


    # Modify input and output depths
    output_depth = 3
    num_channels = [128] * 5
    print(num_channels)
    psrn_noisy_last = 0

    # Adjust loss function
    mse = torch.nn.MSELoss().type(dtype)
    # img_var_list = [np_to_torch(img_np).type(dtype) for img_np in img_np_list]
    # noise_var_list = [np_to_torch(img_mask_np#).type(dtype) for img_mask_np in img_noisy_np_list]

    INPUT = "noise"
    totalupsample = 2**len(num_channels)
    width = int(img_np.shape[1]/totalupsample)
    height = int(img_np.shape[1]/totalupsample)
    shape = [1,num_channels[0], width, height]

    #net_input= get_noise(input_depth, INPUT, img_np.shape[1:]).permute(1, 0, 3, 2).type(dtype) 
    net_input = Variable(torch.zeros(shape))
    net_input.data.uniform_()
    #net_input.data *= 1./10
    net_input = net_input.type(dtype)
    net = decodernw(output_depth, num_channels_up=num_channels , upsample_first=True).type(dtype)
    ## print numbver of parameters in net
    s  = sum([np.prod(list(p.size())) for p in net.parameters()])
    print ('Number of params in decoder: %d' % s)
    ## print nmber of parameter in net_input
    s2  = net_input.shape
    print(s2)

   
    
    print(f"Starting optimization with optimizer '{optim}'")
    if optim =="SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay = weight_decay,momentum = beta)
    elif optim =="ADAM":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)
        print("here")
    elif optim =="SAM":
        base_opt = torch.optim.SGD
        optimizer = SAM(net.parameters(), base_opt, rho=args.reg, adaptive=False, lr=args.lr, weight_decay = weight_decay,momentum = beta) 
          
    i=0
   #[1e-1,1e-2,5e-2],[0.5,0.8] [1e-3 5e-3]
    tot_loss = []
    grad_list = []
    sharp=[]
    psnr_list=[]
    
    def closure_sgd(net_input,img_var,noise_var):
        img_var = np_to_torch(img_var).type(dtype)
        noise_var = np_to_torch(noise_var).type(dtype)
        out = net(net_input)
        blurred_out = convolve_with_gaussian_torch(out, sigma)
        total_loss = mse(blurred_out, noise_var)
        # if optim=="SGD" or  optim=="ADAM":      
        optimizer.zero_grad()     
        total_loss.backward()
        optimizer.step()               
        out_np = out.detach().cpu().numpy()
        img_np = img_var.detach().cpu().numpy()
        img_noisy_np = noise_var.detach().cpu().numpy()
        psnr_gt  = compare_psnr(img_np, out_np)
        #psnr_noisy = compare_psnr(img_noisy_np, out_np)

        return psnr_gt,out_np

    fileid = f'{optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta},reg={reg})'
    outdir = f'data/deblurring/Set14/mask/{ino}/{fileid}/deepdecoder'
    #outdir = f'data/denoising/face/baselines/{ino}/decoder'
    os.makedirs(f'{outdir}', exist_ok=True)
    for j in range(max_steps):
        #optimizer.zero_grad()
        psnr,out = closure_sgd( net_input, img_np, img_noisy_np)
        psnr_noisy = compare_psnr(img_noisy_np, out[0,:,:,:])

        if j%show_every==0 and j!=0:
            if psnr_noisy - psrn_noisy_last < -2: 
                print('Falling back to previous checkpoint.')
                for new_param, net_param in zip(last_net, net.parameters()):
                    net_param.detach().copy_(new_param.cuda())
            else:
                last_net = [x.detach().cpu() for x in net.parameters()]
                psrn_noisy_last = psnr_noisy
            print(f"At step '{j}', psnr is '{psnr}', noisy psnr is '{psnr_noisy}'")
            psnr_list.append(psnr)  
    ##plot and save psnr list in train folder with figure name including ino 
    plt.plot(psnr_list)
    plt.savefig(f'{outdir}/psnr_{ino}.png')
    plt.close()
    ## save the list "psnr" as an npz file and save in the outdir folder
    np.savez(f'{outdir}/psnr_{ino}.npz', psnr=psnr_list)

    output_paths = [
    f"{outdir}/out_{ino}.png",
    f"{outdir}/img_np_{ino}.png",
    f"{outdir}/img_noisy_np_{ino}.png"]  
    print(out.shape, img_np.shape, img_noisy_np.shape)
    images_to_save = [out[0,:,:,:].transpose(1,2,0), img_np.transpose(1,2,0), img_noisy_np.transpose(1,2,0)]
    for path, img in zip(output_paths, images_to_save):
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()
            ## plot the psnr which is a list and multiply the iteration index by showevery to get the x-axis
        plt.plot(psnr)
        plt.title(f'PSNR vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('PSNR')
        plt.savefig(f'{outdir}/psnr_{ino}.png')
        plt.close()      
    

    torch.cuda.empty_cache()
    print("Experiment done") 
            

    plt.plot(psnr_list)
    plt.savefig(f'{outdir}/psnr_{ino}.png')
    plt.close()
    ## save the list "psnr" as an npz file and save in the outdir folder
    np.savez(f'{outdir}/psnr_{ino}.npz', psnr=psnr_list)
    

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
    args = parser.parse_args()
    
    main(images=args.images, lr=args.lr, max_steps=args.max_steps, optim= args.optim,reg=args.reg,sigma = args.sigma, num_layers = args.num_layers, show_every = args.show_every, beta = args.beta, device_id = args.device_id,ino = args.ino, weight_decay = args.decay)
        
    
    
        
        
    
    

