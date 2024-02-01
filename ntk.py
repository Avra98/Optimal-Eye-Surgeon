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

import argparse

def main(images: list, lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2, num_layers: int = 4, show_every: int=1000, device_id: int = 0,beta: float = 0.0,ino : int =0,weight_decay: float = 0.0):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device() 

    def normalize_image(img):
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val)  

    def compare_psnr(img1, img2):
        MSE = np.mean(np.abs(img1-img2)**2)
        psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
        return psnr 
    
    
    img_np_list=[]
    img_noisy_np_list=[]
    noisy_psnr_list=[]
    # train_folder = 'result/Urban100/image_SRF_2/train'
    train_folder = 'data/denoising/Set14'
    train_noisy_folder = 'data/denoising/Set14/train_noisy_{}'.format(sigma)

    os.makedirs(train_noisy_folder, exist_ok=True)

    for i, file_path in enumerate(glob.glob(os.path.join(train_folder, '*.png'))):
        if i == ino:  # we start counting from 0, so the 3rd image is at index 2
            # Get the filename (without extension) for use in messages
            filename = os.path.splitext(os.path.basename(file_path))[0]
            #imsize = -1
            #img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_pil = Image.open(file_path)
            img_pil = resize_and_crop(img_pil, max(img_pil.size))
            img_np = pil_to_np(img_pil)
            print(img_np.shape)
            img_noisy_np = img_np +np.random.normal(scale=sigma, size=img_np.shape)
            img_noisy_np = np.clip(img_noisy_np , 0, 1).astype(np.float32)
            img_np_list.append(img_np)
            img_noisy_np_list.append(img_noisy_np)            
            img_noisy_pil = np_to_pil(img_noisy_np)
            img_noisy_pil.save(os.path.join(train_noisy_folder, filename + '.png'))
            break  # exit the loop

    noisy_psnr = compare_psnr(img_np,img_noisy_np)
    noisy_psnr_list.append(noisy_psnr)
    print(f'Starting vanilla DIP on {ino} using {optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta})')
    print(f"Noisy PSNR is '{noisy_psnr}'")

    # Modify input and output depths
    input_depth = 32   
    output_depth = 3

    # Adjust loss function
    mse = torch.nn.MSELoss().type(dtype)
    # img_var_list = [np_to_torch(img_np).type(dtype) for img_np in img_np_list]
    # noise_var_list = [np_to_torch(img_mask_np).type(dtype) for img_mask_np in img_noisy_np_list]
    INPUT = "noise"      
    net_input= get_noise(input_depth, INPUT, (256,256)).type(dtype) 
    print(img_np.shape[1:])

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
   
    i=0
    tot_loss = []
    grad_list = []
    sharp=[]
    psnr_list=[]
    
    def compute_ntk(net, net_input, device):
        net.to(device)
        net.zero_grad()
        output = net(net_input)        
        output_flat = output.view(-1)
        n = output_flat.size(0)
        # Initialize the NTK matrix
        ntk = torch.zeros((n, n), device=device)        
        # Compute the Jacobian for each element in the flattened output
        for i in range(n):
            net.zero_grad()
            output_flat[i].backward(retain_graph=True)
            grad_i = torch.cat([p.grad.view(-1) for p in net.parameters()])
            ntk[i, :] = grad_i
        # NTK matrix is symmetric, only need to compute half and mirror
        ntk = torch.mm(ntk.t(), ntk)
        return ntk
    
    ntk = compute_ntk(net, net_input, device_id)
    # Compute the eigendecomposition on the device
    eigenvalues, eigenvectors = torch.linalg.eigh(ntk)
    # Get the top eigenvector
    top_eigenvector = eigenvectors[:, -1]  # Assuming the largest eigenvalue is at the last index
    # Reshape the top eigenvector back to the 3 * H * W format
    H, W = net_input.shape[2], net_input.shape[3]  # Assuming net_input is of shape [1, 3, H, W]
    top_eigenvector_image = top_eigenvector.view(3, H, W).detach()
    # Visualize the top eigenvector as an image
    plt.imshow(np.transpose(top_eigenvector_image.cpu().numpy(), (1, 2, 0)))
    plt.title("Top Eigenvector of NTK")
    plt.axis('off')
    plt.savefig('top_eigenvector_ntk.png', bbox_inches='tight', pad_inches=0)  # Save the figure
    plt.close()  # Close the plot to free up memory

    # Visualization of the eigenvalue distribution
    plt.plot(eigenvalues.cpu().numpy())
    plt.title("Eigenvalue Distribution of NTK")
    plt.xlabel('Index')
    plt.ylabel('Eigenvalue')
    plt.savefig('eigenvalue_distribution_ntk.png', bbox_inches='tight')  # Save the figure
    plt.close()  # Close the plot to free up memory
    

    torch.cuda.empty_cache()
    print("Experiment done")           
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")
    
    parser.add_argument("--images", type=str, default = ["Lena512rgb"], help="which image to denoise")
    parser.add_argument("--lr", type=float,  default=1e-2, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=80000, help="the maximum number of gradient steps to train for")
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
        
    
    
        
        
    
    

