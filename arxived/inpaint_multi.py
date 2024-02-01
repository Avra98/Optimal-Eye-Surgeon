
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

def main(images: list, lr: float, max_steps: int, IGR: bool = False, reg: float = 0.0, frac_img: float = 0.2, numlayers: int = 2, device_id: int = 0):
    torch.cuda.set_device(device_id)
    torch.cuda.current_device()

    # Load images and masks
    img_np_list = []
    img_mask_np_list = []

    for image in images:
        imagename = "image_" + str(image) + ".png"
        fname = 'data/denoising/Dataset' + "/" + imagename
        imsize = -1
        img_pil = crop_image(get_image(fname, imsize)[0], d=32)
        img_np = pil_to_np(img_pil)
        img_np = img_np[0, :, :]
        img_mask_np = np.random.binomial(n=1, p=frac_img, size=(img_np.shape[0], img_np.shape[0]))

        img_np_list.append(img_np)
        img_mask_np_list.append(img_mask_np)

    # Modify input and output depths
    input_depth = 3
    output_depth = 1

    # Other parameters remain the same

    # Adjust loss function
    mse = torch.nn.MSELoss().type(dtype)
    img_var_list = [np_to_torch(img_np).type(dtype) for img_np in img_np_list]
    mask_var_list = [np_to_torch(img_mask_np).type(dtype) for img_mask_np in img_mask_np_list]

    def total_loss(out, img_var_list, mask_var_list):
        loss = 0
        for i in range(output_depth):
            loss += mse(out[:, i, :, :] * mask_var_list[i], img_var_list[i] * mask_var_list[i])
        return loss

    # Update PSNR function and optimization loop
    def compare_psnr_multi(img_list, out_list):
        psnr_values = []
        for img1, img2 in zip(img_list, out_list):
            MSE = np.mean(np.abs(img1 - img2) ** 2)
            psnr = 10 * np.log10(np.max(np.abs(img1)) ** 2 / MSE)
            psnr_values.append(psnr)
        return psnr_values

    # Different network inputs for each image instance
    net_input_list = [get_noise(input_depth, INPUT, img_np.shape[0:]) for img_np in img_np_list]

    def closure(j):
        optimizer.zero_grad()
        out = net(torch.cat(net_input_list, dim=0))

        # Calculate total loss
        loss = total_loss(out, img_var_list, mask_var_list)
        loss.backward(create_graph=True, retain_graph=True)

        if IGR:
            grads = 0.0
            for param in net.parameters():
                grads += torch.norm(param.grad) ** 2
            implicit = reg * grads
            implicit.backward()
        else:
            loss.backward()

        with torch.no_grad():
            # Calculate PSNR for each output image
            out_np_list = [out[i, 0, :, :].detach().cpu().numpy() for i in range(batch_size)]
            psnr_gt_list = compare_psnr_multi(img_np_list, out_np_list)
            psnr_list.append(psnr_gt_list)

            if np.mod(j, 10) == 0:
                print(j, psnr_gt_list)


    # Saving results
    psnr = psnr_list.copy()
    mask = mask_loss_list.copy()
    for i, image in enumerate(images):
        np.savez(f"result/inpainting/{image}_{lr}_{reg}_{frac_img}.npz", mask, psnr)
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