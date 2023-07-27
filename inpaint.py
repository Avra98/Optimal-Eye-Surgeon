from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np


from torch import Tensor
from models import *
import torch
import torch.optim
import time
#from skimage.measure import compare_psnr
from scipy.sparse.linalg import LinearOperator, eigsh
from utils.inpainting_utils import * 
import _pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import argparse



def main(images:str, lr:float, max_steps:int, IGR: bool=False, reg: float=0.0, frac_img :float =0.2, numlayers: int = 2, device_id: int=0):
    

    #os.environ['CUDA_VISIBLE_DEVICES'] = 'device_id'
    torch.cuda.set_device(device_id)
    torch.cuda.current_device()
    ##load image and mask
    imagename = "image_"+str(images)+".png"
    fname = 'data/denoising/Dataset'+ "/" +imagename
    imsize =-1
    img_pil = crop_image(get_image(fname, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)  
    img_np = img_np[0,:,:]
    img_mask_np = np.random.binomial(n=1, p=frac_img, size=(img_np.shape[0], img_np.shape[0]))
    

    
    ##specify parameters
    pad = 'reflection'
    OPT_OVER = 'net'
    learning_rate = LR = lr
    #exp_weight = 0.99
    input_depth = 1
    output_depth = 1
    INPUT = 'noise'
    show_every = 500
    num_layers= numlayers

    ## Loss
    mse = torch.nn.MSELoss().type(dtype)
    img_var = np_to_torch(img_np).type(dtype)
    mask_var = np_to_torch(img_mask_np).type(dtype)
    
    def compare_psnr(img1, img2):
        MSE = np.mean(np.abs(img1-img2)**2)
        psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
        #psnr = 10*math.log10(float(1.**2)/MSE)
        return psnr
   
    
    
    num_iter = max_steps

    param_noise_sigma = 2
    reg_noise_std = 1./30

    # weight_decay
    #image_size = img_np.shape[1] * img_np.shape[2]
    weight_decay = 0.0
    #print('%s: %.2e' % (ffname, weight_decay))

    samples = []

    net = skip(
         input_depth, output_depth,
         num_channels_down = [16, 32, 64, 128,128][:num_layers],
         num_channels_up   = [16, 32, 64, 128,128][:num_layers],
         num_channels_skip = [0]*num_layers,
         upsample_mode='nearest',
         downsample_mode='avg',
         need1x1_up = False,
         filter_size_down=3, 
         filter_size_up=3,
         filter_skip_size = 1,
         need_sigmoid=True, 
         need_bias=True, 
         pad='reflection', 
         act_fun='LeakyReLU').type(dtype)

    
    global psnr_gt,psnr_list
    net_input = get_noise(input_depth, INPUT, img_np.shape[0:]).type(dtype)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    # save tmp PSNR for different learning strategies
    psnr_list = [] # psnr between sgld out and gt
    mask_loss_list= [] # loss inside the mask 
    psnr_gt = 0.0
    net_input = net_input_saved
    ## Optimizing 
    print('Starting optimization with SGD')
    optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay = weight_decay)
    
    
    def closure_sgd(j):
        out = net(net_input)
        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward(create_graph=True, retain_graph=True)
        
        if (IGR==True):    
            total_loss.backward(create_graph=True, retain_graph=True)
            grads =0.0
            for param in net.parameters():
                grads += torch.norm(param.grad)**2
            implicit = reg*grads
            implicit.backward()
        else:
            total_loss.backward()
            

        with torch.no_grad():
            mask_com = torch.logical_not(mask_var).cpu().detach()
            mask_loss = mse(out.cpu().detach() * mask_com, img_var.cpu().detach() * mask_com)
            mask_loss_list.append(mask_loss)

            out_np = out.detach().cpu().numpy()[0]
            psnr_gt  = compare_psnr(img_np, out_np)
            psnr_list.append(psnr_gt) 

            if np.mod(j, 10) == 0:
                #plt.imshow(out[0,0,:,:].detach().cpu().numpy(),cmap="gray")
                print(j,psnr_gt)
                
                #plt.show()   


    for j in range(num_iter):
        optimizer.zero_grad()
        closure_sgd(j)    
        optimizer.step()
         
    psnr= psnr_list.copy()
    mask= mask_loss_list.copy()
    np.savez("result/inpainting/"+"_"+str(images)+"_"+str(lr)+"_"+str(reg)+"_"+str(frac_img)+".npz",mask,psnr)
    torch.cuda.empty_cache()
    print("Experiment done")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image inpainiting using DIP")
    
    parser.add_argument("images", type=str, help="which image to inpaint")
    parser.add_argument("lr", type=float, help="the learning rate")
    parser.add_argument("max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--IGR", type=bool, default=False, help="if sgd-igr, gd-igr, or label-wised-sgd-igr")
    parser.add_argument("--reg", type=float, default=3e-4, help="if regularization strength of igr")
    parser.add_argument("--frac_img", type=float, default=0.3, help="fraction of image observed")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--device_id", type=int, default=0, help="specify which gpu")
    args = parser.parse_args()
    
    main(images=args.images, lr=args.lr, max_steps=args.max_steps, IGR=args.IGR, reg=args.reg,frac_img = args.frac_img, numlayers = args.num_layers, device_id = args.device_id)    
    
    
    
    
    
    
    
    
    
