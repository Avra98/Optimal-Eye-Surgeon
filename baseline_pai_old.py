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
from DIP_quant.utils.quant import *
from imp import *       
from models.cnn import cnn
import torch
import torch.optim
import time
#from skimage.measure import compare_psnr
from utils.inpainting_utils import * 
import pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sam import SAM

import argparse

def main(images: list, lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2,
          num_layers: int = 4, show_every: int=1000, device_id: int = 0,beta: float = 0.0,
          ino : int =0,weight_decay: float = 0.0, sparse: float = 0.5,prune_type: str = "rand_global"):

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
    print(f'Starting random mask  training on {ino} using {optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta})')
    print(f"Noisy PSNR is '{noisy_psnr}'")
    

    # Modify input and output depths
    input_depth = 32    
    output_depth = 3
    

    # Adjust loss function
    
    mse = torch.nn.MSELoss().type(dtype)
    # img_var_list = [np_to_torch(img_np).type(dtype) for img_np in img_np_list]
    # noise_var_list = [np_to_torch(img_mask_np).type(dtype) for img_mask_np in img_noisy_np_list]

    INPUT = "noise"
        
    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)
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
    
    mask=  get_pruning_mask(net)  
    if prune_type == "rand_global":
        mask= prune_random_global(net,mask,sparse)  
    elif prune_type == "rand_local":
        mask= prune_random_local(net,mask,sparse)
    elif prune_type == "mag_global":
        mask= prune_magnitude_global(net,mask,sparse)     
    elif prune_type == "snip":
        mask= snip_prune(net,mask,net_input,img_np, img_noisy_np,sparse)
    elif prune_type == "grasp":
        mask= grasp_prune(net,mask,net_input,img_np, img_noisy_np,sparse)  
    elif prune_type == "synflow":
        mask= synflow_prune(net,mask,net_input,sparse)      
    elif prune_type == "snip_local":
        mask= snip_prune_local(net,mask,net_input,img_np, img_noisy_np,sparse)
    elif prune_type == "grasp_local":
        mask= grasp_prune_local(net,mask,net_input,img_np, img_noisy_np,sparse)  
    elif prune_type == "synflow_local":
        mask= synflow_prune_local(net,mask,net_input,sparse)  
        # base_opt = torch.optim.SGD
        # optimizer = SAM(net.parameters(), base_opt, rho=args.reg, adaptive=False, lr=args.lr, weight_decay = weight_decay,momentum = beta) 

        
    psnr,out =  train_sparse(net,net_input,mask, img_np, img_noisy_np,max_step=args.max_steps,show_every=args.show_every,device=args.device_id)
    ## save the output image and psnr in the outdir folder

    #fileid = f'{optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta})'
    #outdir = f'data/denoising/Set14/mask/{ino}/pai/{prune_type}/sparse_{sparse}/{sigma}'
    #outdir = f'data/denoising/face/mask/{ino}/pai/{prune_type}'
    outdir = f'data/denoising/Dataset/mask/{ino}/pai/{prune_type}_{sparse}'
    print(f"Output directory: {outdir}")
    os.makedirs(f'{outdir}', exist_ok=True)

    with torch.no_grad():

        out_np = out
        img_var = np_to_torch(img_np)
        img_np = img_var.detach().cpu().numpy()
        psnr_gt  = compare_psnr(img_np, out_np)
        print("PSNR of output image is: ", psnr_gt)    
        ## save the list "psnr" as an npz file and save in the outdir folder
        np.savez(f'{outdir}/psnr_{ino}.npz', psnr=psnr)
        

        output_paths = [
        f"{outdir}/out_{ino}.png",
        f"{outdir}/img_np_{ino}.png",
        f"{outdir}/img_noisy_np_{ino}.png"]
            
        print(out_np.shape, img_np.shape, img_noisy_np.shape)
        images_to_save = [out_np.transpose(1,2,0), img_np[0,:,:,:].transpose(1,2,0), img_noisy_np.transpose(1,2,0)]
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
            plt.savefig(f'{outdir}/psnr_{ino}_{sparse}_{prune_type}_{sigma}.png')
            plt.close()

            # plt.hist(quant_weight, bins=50, alpha=0.5, label='Quantized Weights')
            # plt.title(f'Quantized Weights Histogram')
            # plt.xlabel('Quantized Weight Values')
            # plt.ylabel('Frequency')
            # plt.legend()

            # # Save the histogram plot in the same directory as other figures
            # plt.savefig(f'{outdir}/out_images/quant_weight_histogram_{ino}.png') 
            # plt.close()  
                            


    torch.cuda.empty_cache()
    print("Experiment done")           
    #plot_psnr(psnr_lists)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")
    
    parser.add_argument("--images", type=str, default = ["Lena512rgb"], help="which image to denoise")
    parser.add_argument("--lr", type=float,  default=1e-2, help="the learning rate")
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
    parser.add_argument("--ino", type=int, default=4, help="image index ")
    parser.add_argument("--sparse", type=float, default=0.5, help="sparse perecentage")
    parser.add_argument("--prune_type", type=str, default="rand_global", help="sparse perecentage")

    args = parser.parse_args()
    
    main(images=args.images, lr=args.lr, max_steps=args.max_steps, 
         optim= args.optim,reg=args.reg,sigma = args.sigma, num_layers = args.num_layers, 
         show_every = args.show_every, beta = args.beta, device_id = args.device_id,ino = args.ino,
         weight_decay = args.decay,sparse = args.sparse,prune_type=args.prune_type)
        
    
    
        
