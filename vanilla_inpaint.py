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

def main(images: list, lr: float, max_steps: int, optim: str, reg: float = 0.0, 
         p: float = 0.2, num_layers: int = 4, show_every: int=1000, 
         device_id: int = 0,beta: float = 0.0,ino : int =0,weight_decay: float = 0.0):

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
    train_folder = 'data/inpainting/Set14'
    train_mask_folder = 'data/inpainting/Set14/train_inpaint_{}'.format(p)

    os.makedirs(train_mask_folder, exist_ok=True)

    for i, file_path in enumerate(glob.glob(os.path.join(train_folder, '*.png'))):
        if i == ino:  # we start counting from 0, so the 3rd image is at index 2
            # Get the filename (without extension) for use in messages
            filename = os.path.splitext(os.path.basename(file_path))[0]
            imsize = -1
            img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_pil = Image.open(file_path)
            img_pil = resize_and_crop(img_pil, max(img_pil.size))
            img_np = pil_to_np(img_pil)
            print(img_np.shape)
            _,img_mask_np  = get_bernoulli_mask(img_pil, p)
            print(img_mask_np.shape)
            img_masked  = img_np * img_mask_np
            mask_var    = np_to_torch(img_mask_np).type(dtype)
            break  # exit the loop
    noisy_psnr = compare_psnr(img_np,img_masked)
    noisy_psnr_list.append(noisy_psnr)
    print(f'Starting vanilla DIP on {ino} using {optim}(p={p},lr={lr},decay={weight_decay},beta={beta})')
    print(f"Noisy PSNR is '{noisy_psnr}'")
        

    # Modify input and output depths
    input_depth = 32   
    output_depth = 3

    # Adjust loss function
    mse = torch.nn.MSELoss().type(dtype)
    # img_var_list = [np_to_torch(img_np).type(dtype) for img_np in img_np_list]
    # noise_var_list = [np_to_torch(img_mask_np).type(dtype) for img_mask_np in img_noisy_np_list]

    INPUT = "noise"
        
    net_input= get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype) 
    # print("input dim:", net_input.shape) [1, 3, 256, 256]
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

   
    
    print(f"Starting optimization with optimizer '{optim}'")
    if optim =="SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay = weight_decay,momentum = beta)
    elif optim =="ADAM":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)
    elif optim =="SAM":
        base_opt = torch.optim.SGD
        optimizer = SAM(net.parameters(), base_opt, rho=args.reg, adaptive=False, lr=args.lr, weight_decay = weight_decay,momentum = beta) 
          
    i=0
   #[1e-1,1e-2,5e-2],[0.5,0.8] [1e-3 5e-3]
    tot_loss = []
    grad_list = []
    sharp=[]
    psnr_list=[]

    def closure_sgd(net_input,img_var,mask_var):
        img_var = np_to_torch(img_var).type(dtype)
        #mask_var = np_to_torch(mask_var).type(dtype)
        out = net(net_input)
        # with torch.no_grad():
        #     print(net_input.shape,img_var.shape,out.shape)
        total_loss = mse(out*mask_var, img_var*mask_var)
        if optim=="SGD" or  optim=="ADAM":      
            optimizer.zero_grad()     
            total_loss.backward()
            optimizer.step()              
        out_np = out.detach().cpu().numpy()[0]
        img_np = img_var.detach().cpu().numpy()
        psnr_gt  = compare_psnr(img_np, out_np)
        return psnr_gt,out_np

    outdir = f'data/inpainting/Set14/{ino}/{p}'
    os.makedirs(f'{outdir}/vanilla', exist_ok=True)
    for j in range(max_steps):
        #optimizer.zero_grad()
        psnr,out = closure_sgd( net_input, img_np, mask_var)
        #print(out.shape)

        if j%show_every==0 and j!=0:
            print(f"At step '{j}', psnr is '{psnr}'")
            psnr_list.append(psnr)  
    ##plot and save psnr list in train folder with figure name including ino 
    plt.plot(psnr_list)
    plt.savefig(f'{outdir}/psnr_{ino}.png')
    plt.close()
    ## save the list "psnr" as an npz file and save in the outdir folder
    np.savez(f'{outdir}/psnr_{ino}.npz', psnr=psnr_list)
    ##  imshow and imsave out_np in train folder with figure name including ino
    # Saving the output image
    fig, ax = plt.subplots()
    ax.imshow(out.transpose(1, 2, 0))
    ax.axis('off')  # Turn off axis
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplot parameters to make the image fit the figure area.
    plt.savefig(f'{outdir}/vanilla/out_{ino}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Saving the masked image
    fig, ax = plt.subplots()
    ax.imshow(img_masked.transpose(1, 2, 0))
    ax.axis('off')  # Turn off axis
    #plt.title(f'Masked Image {ino}')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Adjust subplot parameters to make the image fit the figure area.
    plt.savefig(f'{outdir}/vanilla/masked_{ino}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    

    torch.cuda.empty_cache()
    print("Experiment done")           
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")
    
    parser.add_argument("--images", type=str, default = ["Lena512rgb"], help="which image to denoise")
    parser.add_argument("--lr", type=float,  default=1e-3, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=40000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--optim", type=str, default="ADAM", help="which optimizer")
    #parser.add_argument("--IGR", type=str, default="Normal", help="true if SAM ")
    parser.add_argument("--reg", type=float, default=0.05, help="if regularization strength of igr")
    parser.add_argument("--p", type=float, default=0.5, help="mask level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show_every")
    parser.add_argument("--device_id", type=int, default=1, help="specify which gpu")
    parser.add_argument("--beta", type=float, default=0, help="momentum for sgd ")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--ino", type=int, default=0, help="image index ")
    args = parser.parse_args()
    
    main(images=args.images, lr=args.lr, max_steps=args.max_steps, optim= args.optim,reg=args.reg,p = args.p, num_layers = args.num_layers, show_every = args.show_every, beta = args.beta, device_id = args.device_id,ino = args.ino, weight_decay = args.decay)
        
    
    
        
        
    
    

