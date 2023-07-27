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
from datetime import datetime
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

# Load the model architecture
model = skip(
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
        act_fun='LeakyReLU').type(dtype)    # replace this with the initialization of your model

# Load the state dict previously saved
model.load_state_dict(torch.load('model_sigma_0.2_optim_SGD_reg_0.05.pth'))

# Don't forget to set the model to evaluation mode if you're doing inference
model.eval()

# Load new images, add noise, and try to denoise them
test_folder = 'path_to_your_test_folder'  # replace with your test folder path
test_noisy_folder = 'path_to_your_noisy_test_folder'  # replace with your noisy test folder path

img_np_list=[]
img_noisy_np_list=[]

for i, file_path in enumerate(glob.glob(os.path.join(test_folder, '*.png'))):
    filename = os.path.splitext(os.path.basename(file_path))[0]
    imsize = -1
    img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
    img_np = pil_to_np(img_pil)
    img_np = img_np[0, :, :]
    
    img_noisy_np = img_np + np.random.normal(scale=sigma, size=img_np.shape)
    img_noisy_np = np.clip(img_noisy_np , 0, 1).astype(np.float32)
    
    print(np.max(img_np), np.min(img_np))
    img_np_list.append(img_np)
    img_noisy_np_list.append(img_noisy_np)
    
    print(f"Noisy PSNR is '{compare_psnr(img_np,img_noisy_np)}'")
    img_noisy_pil = np_to_pil(img_noisy_np)
    img_noisy_pil.save(os.path.join(test_noisy_folder, filename + '.png'))

# Set requires_grad to True for net inputs
net_input_list = [get_noise(input_depth, INPUT, img_np.shape[0:]).type(dtype).requires_grad_() for img_np in img_np_list]

optimizer = torch.optim.Adam([{'params': net_input}], lr=lr)

# Optimization process over the net_input
for j in range(max_steps):
    for m in range(len(img_np_list)):
        optimizer.zero_grad()
        
        img_var = np_to_torch(img_np_list[m]).type(dtype)
        noise_var = np_to_torch(img_noisy_np_list[m]).type(dtype)

        out = net(net_input_list[m])
        total_loss = mse(out, noise_var)
        
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss = mse(out.detach().cpu(), img_var.detach().cpu())
                
            out_np = out.detach().cpu().numpy()[0]
            img_np = img_var.detach().cpu().numpy()
            psnr_gt  = compare_psnr(img_np, out_np)
            psnr_lists[m].append(psnr_gt)

        print(f"At step '{j}', for image '{m}', psnr is '{psnr_gt}'")
