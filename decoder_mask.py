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

import argparse

def main(images: list, lr: float, max_steps: int, k:int, optim: str, reg: float = 0.0, sigma: float = 0.2,
          num_layers: int = 4, show_every: int=1000, device_id: int = 0,beta: float = 0.0,
          ino : int =0,weight_decay: float = 0.0, mask_opt: str = "single", noise_steps: int = 80000,kl: float = 1e-5,prior_sigma: float = 0.0):

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
    print(f'Starting vanilla DIP on {ino} using {optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta})')
    print(f"Noisy PSNR is '{noisy_psnr}'")
    
   # Modify input and output depths
    output_depth = 3
    num_channels = [128] * args.k
    print(num_channels)

    # Adjust loss function
    mse = torch.nn.MSELoss().type(dtype)

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
    print (s2)
     
    print(f"Starting optimization with optimizer '{optim}'")
    if optim =="SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay = weight_decay,momentum = beta)
    elif optim =="ADAM":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)
    elif optim =="SAM":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay = weight_decay,momentum = beta)


    fileid = f'{optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta})'
    outdir = f'data/denoising/Set14/decoder_mask/{ino}/{fileid}/{mask_opt}/{prior_sigma}/{kl}'
    print(f"Output directory: {outdir}")
    os.makedirs(f'{outdir}/out_images/', exist_ok=True)

    m=0
    #p,quant_loss = learn_quantization_probabilities_dip(net,net_input, img_np, img_noisy_np, num_steps, lr,ino, q=2,kl=args.kl,prior_sigma=args.prior_sigma)
    p,quant_loss = learn_quantization_probabilities_dip(net,net_input, img_np, img_noisy_np, max_steps, lr,ino, q=2,kl=args.kl,prior_sigma=args.prior_sigma)
            ## save p in the outdir folder
    mask = make_mask(p)        
    masked_model = mask_network(mask,net)  
    ## save masked model and net_input_list in the outdir folder
    with open(f'{outdir}/masked_model_{ino}.pkl', 'wb') as f:
        cPickle.dump(masked_model, f)
    with open(f'{outdir}/net_input_list_{ino}.pkl', 'wb') as f:
        cPickle.dump(net_input, f)
    with open(f'{outdir}/mask_{ino}.pkl', 'wb') as f:
       cPickle.dump(mask, f)
    #     
                
    with torch.no_grad():
        if mask_opt=='single':          
            out = draw_one_mask(p, net,net_input)
        elif mask_opt=='multiple':
            out = draw_multiple_masks(p, net,net_input)
        else:
            print("here")
            out = deterministic_rounding(p, net,net_input)


        out_np = out.detach().cpu().numpy()[0]
        img_var = np_to_torch(img_np)
        img_np = img_var.detach().cpu().numpy()
        psnr_gt  = compare_psnr(img_np, out_np)
        print("PSNR of output image is: ", psnr_gt)        

        output_paths = [
        f"{outdir}/out_images/out_{ino}.png",
        f"{outdir}/out_images/img_np_{ino}.png",
        f"{outdir}/out_images/img_noisy_np_{ino}.png"]
            
        # print(out_np.shape, img_np.shape, img_noisy_np.shape)
        images_to_save = [out_np.transpose(1,2,0), img_np[0,:,:,:].transpose(1,2,0), img_noisy_np.transpose(1,2,0)]
        for path, img in zip(output_paths, images_to_save):
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()
            # plt.hist(quant_weight, bins=50, alpha=0.5, label='Quantized Weights')
            # plt.title(f'Quantized Weights Histogram')
            # plt.xlabel('Quantized Weight Values')
            # plt.ylabel('Frequency')
            # plt.legend()

            # # Save the histogram plot in the same directory as other figures
            # plt.savefig(f'{outdir}/out_images/quant_weight_histogram_{ino}.png') 
            # plt.close()  

        plt.plot(range(0, len(quant_loss) * 1000, 1000), quant_loss, marker='o', linestyle='-')
        plt.title('Quantization Loss Over Training Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Quantization Loss')
        plt.grid(True)

            # Save the quant_loss plot in the same directory as other figures
        plt.savefig(f'{outdir}/out_images/qquant_loss_{ino}.png') 
                            


    torch.cuda.empty_cache()
    print("Experiment done")           
    def plot_psnr(psnr_lists):
            # filedir = f"{outdir}"
            # os.makedirs(filedir)
            for i, psnr_list in enumerate(psnr_lists):
                plt.figure(figsize=(10, 5))
                plt.plot(psnr_list)
                plt.title(f"Image {ino} (vanilla dip)")
                plt.xlabel("Iteration")
                plt.ylabel("PSNR")
                plt.grid(True)
                plt.axhline(y=noisy_psnr_list[i])
                plt.savefig(f"{outdir}/psnr_{fileid}.png") 

    #plot_psnr(psnr_lists)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")   
    parser.add_argument("--images", type=str, default = ["Lena512rgb"], help="which image to denoise")
    parser.add_argument("--lr", type=float,  default=1e-2, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=80000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--optim", type=str, default="SAM", help="which optimizer")
    #parser.add_argument("--IGR", type=str, default="Normal", help="true if SAM ")
    parser.add_argument("--reg", type=float, default=0.05, help="if regularization strength of igr")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise-level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show_every")
    parser.add_argument("--device_id", type=int, default=0, help="specify which gpu")
    parser.add_argument("--beta", type=float, default=0, help="momentum for sgd ")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--ino", type=int, default=0, help="image index ")
    parser.add_argument("--mask_opt", type=str, default="det", help="mask type")
    parser.add_argument("--noise_steps", type=int, default=80000, help="numvere of steps for noise")
    parser.add_argument("--kl", type=float, default=1e-9, help="regularization strength of kl")
    parser.add_argument("--prior_sigma", type=float, default=0.0, help="prior mean")
    parser.add_argument("--k", type=int, default=5, help="number of channels ")
    args = parser.parse_args()
    
    main(images=args.images, lr=args.lr, max_steps=args.max_steps, k=args.k,
         optim= args.optim,reg=args.reg,sigma = args.sigma, num_layers = args.num_layers, 
         show_every = args.show_every, beta = args.beta, device_id = args.device_id,ino = args.ino,
         weight_decay = args.decay,mask_opt = args.mask_opt, noise_steps = args.noise_steps,kl = args.kl,prior_sigma=args.prior_sigma)
        
    
    
