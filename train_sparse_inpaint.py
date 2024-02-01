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
from PIL import Image
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

def inverse_sigmoid(x):
    return torch.log(torch.tensor(x) / torch.tensor(1 - x))

def main(images: list, lr: float, max_steps: int, optim: str, reg: float = 0.0, p: float = 0.2,
          num_layers: int = 4, show_every: int=1000, device_id: int = 0,beta: float = 0.0,
          ino_img : int =0, ino_net :int=0, weight_decay: float = 0.0, mask_opt: str = "det", noise_steps: int = 80000,
          kl: float = 1e-5,prior_sigma: float = 0.0, sparsity: float = 0.05):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device() 
    prior_sigma=inverse_sigmoid(args.sparsity)

    def normalize_image(img):
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val)  

    def compare_psnr(img1, img2):
        MSE = np.mean(np.abs(img1-img2)**2)
        psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
        return psnr 
    
    def add_noise(model, noise_scale):
        for n in [x for x in model.parameters() if len(x.size()) == 4]:
            ran = torch.rand(n.shape)
            noise_tensor = ran * noise_scale * args.lr  # Renamed variable
            noise_tensor = noise_tensor.type(dtype)
            n.data = n.data + noise_tensor    

    
    img_np_list=[]
    img_noisy_np_list=[]
    noisy_psnr_list=[]
    train_folder = 'data/inpainting/Set14'
    train_mask_folder = 'data/inpainting/Set14/train_inpaint_{}'.format(p)

    os.makedirs(train_mask_folder, exist_ok=True)

    for i, file_path in enumerate(glob.glob(os.path.join(train_folder, '*.png'))):
        if i == ino_img:  # we start counting from 0, so the 3rd image is at index 2
            # Get the filename (without extension) for use in messages
            filename = os.path.splitext(os.path.basename(file_path))[0]           
            # Print the filename
            print(f"Processing image: {filename}")
            imsize = -1
            img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_pil = Image.open(file_path)
            img_pil = resize_and_crop(img_pil, max(img_pil.size))
            img_np = pil_to_np(img_pil)
            print(img_np.shape)
            _, img_mask_np = get_bernoulli_mask(img_pil, p)
            print(img_mask_np.shape)
            img_masked  = img_np * img_mask_np
            mask_var = np_to_torch(img_mask_np).type(dtype)
            break  # exit the loop

    noisy_psnr = compare_psnr(img_np,img_masked)
    noisy_psnr_list.append(noisy_psnr)
    #print(f'Starting vanilla DIP on {ino} using {optim}(sigma=0.1,lr={lr},decay={weight_decay},beta={beta})')
    #print(f"Noisy PSNR is '{noisy_psnr}'")
    

    # Modify input and output depths
    input_depth = 32    
    output_depth = 3
    num_steps = args.noise_steps
    

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

    
    print(f"Starting optimization with optimizer '{optim}'")
    if optim =="SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay = weight_decay,momentum = beta)
    elif optim =="ADAM":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)
    elif optim =="SAM":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay = weight_decay,momentum = beta)
        # base_opt = torch.optim.SGD
        # optimizer = SAM(net.parameters(), base_opt, rho=args.reg, adaptive=False, lr=args.lr, weight_decay = weight_decay,momentum = beta) 
          
    # fileid = f'{optim}(sigma=0.1,lr={lr},decay={weight_decay},beta={beta})'
    # outdir = f'data/denoising/Set14/mask/{ino_net}/{fileid}/{mask_opt}/{prior_sigma}/{kl}'
    # print(f"Output directory where image taken from: {outdir}")
    #outdir = f'data/inpainting/Set14/mask/{ino_img}/0.5/{mask_opt}/{prior_sigma}/{kl}'
    #print(f"Output directory where net taken from: {outdir}")
    ## if it created the directory then send a print statement
    #outdir_target = f'data/inpainting/Set14/transfer_inpaint/{p}/{ino_img}/{ino_img}/{prior_sigma}'
    outdir = f'data/inpainting/Set14/mask/{ino_img}/sparsity/0.5/det/{sparsity}/{kl}'
    os.makedirs(f'{outdir}', exist_ok=True)
    print(f"Output directory where results stored: {outdir}")

    m=0
    # load masked model and net_input_list from the outdir folder
    with open(f'{outdir}/masked_model_{ino_img}.pkl', 'rb') as f:
        masked_model = cPickle.load(f)
    with open(f'{outdir}/net_input_list_{ino_img}.pkl', 'rb') as f:
        net_input_list = cPickle.load(f)
    # load the saved mask
    with open(f'{outdir}/mask_{ino_img}.pkl', 'rb') as f:
        mask = cPickle.load(f)
    #psnr,out =  train_sparse(masked_model,net_input_list,mask, img_np, img_noisy_np,max_step=args.max_steps,show_every=args.show_every,device=args.device_id)
    psnr,out =  train_sparse_inpaint(masked_model,net_input_list,mask, img_np, 
                                     mask_var,max_step=args.max_steps,
                                     show_every=args.show_every,device=args.device_id)
    ## save the output image and psnr in the outdir folder
           

    with torch.no_grad():

        out_np = out[0,:,:,:]
        img_var = np_to_torch(img_np)
        img_np = img_var.detach().cpu().numpy()
        psnr_gt  = compare_psnr(img_np, out_np)
        print("PSNR of output image is: ", psnr_gt)    
        ## save the list "psnr" as an npz file and save in the outdir folder
        np.savez(f'{outdir}/psnr_{ino_img}.npz', psnr=psnr)
        

        output_paths = [
        f"{outdir}/out_{ino_img}.png",
        f"{outdir}/img_noisy_np_{ino_img}.png"]
            
        print(out_np.shape, img_np.shape)
        images_to_save = [out_np.transpose(1,2,0), img_masked.transpose(1,2,0)]
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
            plt.savefig(f'{outdir}/psnr_{ino_img}.png')
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
    parser.add_argument("--optim", type=str, default="SAM", help="which optimizer")
    #parser.add_argument("--IGR", type=str, default="Normal", help="true if SAM ")
    parser.add_argument("--reg", type=float, default=0.05, help="if regularization strength of igr")
    parser.add_argument("--p", type=float, default=0.5, help="masking probability")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show_every")
    parser.add_argument("--device_id", type=int, default=0, help="specify which gpu")
    parser.add_argument("--beta", type=float, default=0, help="momentum for sgd ")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--ino_img", type=int, default=0, help="image index ")
    parser.add_argument("--ino_net", type=int, default=0, help="sparse network index")
    parser.add_argument("--mask_opt", type=str, default="det", help="mask type")
    parser.add_argument("--noise_steps", type=int, default=80000, help="numvere of steps for noise")
    parser.add_argument("--kl", type=float, default=1e-9, help="regularization strength of kl")
    parser.add_argument("--prior_sigma", type=float, default=-1.3, help="prior mean")
    parser.add_argument("--sparsity", type=float, default=0.0, help="prior mean")
    args = parser.parse_args()
    
    main(images=args.images, lr=args.lr, max_steps=args.max_steps, 
         optim= args.optim,reg=args.reg,p = args.p, num_layers = args.num_layers, 
         show_every = args.show_every, beta = args.beta, device_id = args.device_id,ino_img = args.ino_img,
         ino_net = args.ino_net,weight_decay = args.decay,mask_opt = args.mask_opt, noise_steps = args.noise_steps,
         kl = args.kl,prior_sigma=args.prior_sigma,sparsity=args.sparsity)
        
    
    
    
