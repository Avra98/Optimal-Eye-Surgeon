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
import argparse


num_layers=6
output_depth=1
input_depth=3
sigma=0.1
mse = torch.nn.MSELoss().type(dtype)

def compare_psnr(img1, img2):
    MSE = np.mean(np.abs(img1-img2)**2)
    psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
    return psnr    

# Load the model architecture
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
        act_fun='LeakyReLU').type(dtype)    # replace this with the initialization of your model

# Load the state dict previously saved
#net.load_state_dict(torch.load('model_optim-SAM_sigma-0.1_2023-08-07_21-45-28.pth'))
#net.load_state_dict(torch.load('model_optim-SGD_sigma-0.1_2023-08-07_18-03-22.pth'))
#net.load_state_dict(torch.load('model_optim-SAM_sigma-0.05_2023-08-07_21-01-54.pth'))
# net.load_state_dict(torch.load('model_optim-SGD_sigma-0.05_2023-08-08_00-28-55.pth'))
# net.load_state_dict(torch.load('model_optim-SAM_sigma-0.05_2023-09-22_05-37-48.pth'))
# Don't forget to set the model to evaluation mode if you're doing inference
def main(
        model: str,
        ino: int,
        device_id: int):
    torch.cuda.set_device(device_id)
    torch.cuda.current_device()
    typedir = 'result/Urban100/minibatch/train'
    # model = 'SGD(batchsize=10,training=100,sigma=0.1,lr=0.08,decay=0.0,beta=0.0)'
    state_files = glob.glob(f'{typedir}/{model}/*.pth')
    if len(state_files) == 0:
        print(f'No state files found for {model}')
        sys.exit(1)
    elif len(state_files) > 1:
        # ask user to select one
        print(f'Multiple state files found for {model}')
        for i, state_file in enumerate(state_files):
            print(f'[{i}] {state_file}')
        while True:
            try:
                i = int(input('Select one: '))
                if i < 0 or i >= len(state_files):
                    raise ValueError()
                break
            except ValueError:
                print('Invalid selection')
        state_file = state_files[i]
    else:
        state_file = state_files[0]

    net.load_state_dict(torch.load(state_file))
    net.eval()
    net.requires_grad_(False)

    max_steps=70000

    # Load new images, add noise, and try to denoise them
    test_folder = 'result/Urban100/image_SRF_2/test' # replace with your test folder path
    test_noisy_folder = 'result/Urban100/image_SRF_2/test_noisy_0.1'  # replace with your noisy test folder path
    # test_folder = 'result/Urban100/image_SRF_2/train' # replace with your test folder path
    # test_noisy_folder = 'result/Urban100/image_SRF_2/train_noisy_0.1'  # replace with your noisy test folder path

    img_np_list = []
    img_noisy_np_list = []
    psnr_list = []

    for i, file_path in enumerate(glob.glob(os.path.join(test_folder, '*.png'))):
        if i==ino:
            # Get the filename (without extension) for use in messages
            filename = os.path.splitext(os.path.basename(file_path))[0]
            imsize = -1
            img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_np = pil_to_np(img_pil)
            img_np = img_np[0, :, :]
            img_noisy_np = img_np +np.random.normal(scale=sigma, size=img_np.shape)
            img_noisy_np = np.clip(img_noisy_np , 0, 1).astype(np.float32)

            noisy_psnr = compare_psnr(img_np,img_noisy_np)

            img_np_list.append(img_np)
            img_noisy_np_list.append(img_noisy_np)
                
            img_noisy_pil = np_to_pil(img_noisy_np)
            break
            #img_noisy_pil.save(os.path.join(train_noisy_folder, filename + '.png'))

    print(img_np.shape)
    # Set requires_grad to True for net inputs
    net_input = get_noise(input_depth, "noise", img_np.shape[0:]).type(dtype).requires_grad_(True)

    #print(net_input.shape,img_np.shape)
    print(len(parameters_to_vector(net.parameters())))
    print(len(parameters_to_vector(net_input)))
        
    # optimizer = torch.optim.Adam([{'params': net_input}], lr=1e-3)
    optimizer = SAM(net.parameters(), torch.optim.SGD, adaptive=False, lr=1e-3) 
    img_var = np_to_torch(img_np).type(dtype)
    noise_var = np_to_torch(img_noisy_np).type(dtype)

    # Optimization process over the net_input
    for j in range(max_steps):
        optimizer.zero_grad()        
        out = net(net_input)
        total_loss = mse(out, noise_var)       
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss = mse(out.detach().cpu(), img_var.detach().cpu())
                    
            out_np = out.detach().cpu().numpy()[0]
            img_np = img_var.detach().cpu().numpy()
            psnr_gt  = compare_psnr(img_np, out_np)
            psnr_list.append(psnr_gt)

            if j % 500 == 0:
                #for name, param in net.named_parameters():
            #     print(name,param.data[0])

                # os.makedirs('result/Urban100/out_test', exist_ok=True)
                # plt.imsave(f'result/Urban100/out_test/out_image_{ino}_{j}.png', img_np, cmap=cm.gray)
                print(f"At step '{j}', psnr is '{psnr_gt}'")

    def plot_psnr(psnr_list, outdir, noisy_psnr):
        filedir = f'{outdir}/psnr'
        os.makedirs(filedir, exist_ok=True)

        plt.figure(figsize=(10, 5))
        plt.plot(psnr_list, label=f"PSNR Image {ino}")
        plt.title(f"PSNR over Iterations for Image {ino}")
        plt.xlabel("Iteration")
        plt.ylabel("PSNR")
        plt.grid(True)
        plt.axhline(y=noisy_psnr, color='r', linestyle='--', label="Noisy PSNR")
        plt.legend()
        plt.savefig(f"{filedir}/psnr_image_{ino}.png")
        plt.close()

    outdir = f"{'/'.join(typedir.split('/')[:-1])}/test/{model}"
    os.makedirs(outdir, exist_ok=True)
    plot_psnr(psnr_list, outdir, noisy_psnr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")

    parser.add_argument("--model", type=str)
    parser.add_argument("--ino", type=int)
    parser.add_argument("--device_id", type=int)
    args = parser.parse_args()
    
    main(model=args.model,
         ino=args.ino,
         device_id=args.device_id)