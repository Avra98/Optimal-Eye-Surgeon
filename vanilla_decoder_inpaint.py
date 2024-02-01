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
from ptflops import get_model_complexity_info
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
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

import argparse

def main(images: list, lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2, num_layers: int = 4, show_every: int=1000, device_id: int = 0,beta: float = 0.0,ino : int =0,weight_decay: float = 0.0):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device() 

    def normalize_image(img):
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val)  

    # def compare_psnr(img1, img2):
    #     MSE = np.mean(np.abs(img1-img2)**2)
    #     psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
    #     return psnr 
    

    # def compute_svd(conv_layer):
    #     weights = conv_layer.weight.data
    #     if len(weights.shape) == 4:  # Convolutional layers have 4-dimensional weights
    #         weights = weights.view(weights.size(0), -1)
    #     _, s, _ = torch.svd(weights)
    #     return s   
    
    
    img_np_list=[]
    img_noisy_np_list=[]
    noisy_psnr_list=[]
    # train_folder = 'result/Urban100/image_SRF_2/train'
    train_folder = 'data/denoising/Set14'
    train_noisy_folder = 'data/denoising/Set14/train_noisy_{}'.format(sigma)

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
        if i == ino:  # we start counting from 0, so the 3rd image is at index 2
            # Get the filename (without extension) for use in messages
            filename = os.path.splitext(os.path.basename(file_path))[0]
            imsize = -1
            #img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_pil = Image.open(file_path)
            #img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_pil = resize_and_crop(img_pil, max(img_pil.size))
            img_np = pil_to_np(img_pil)
            img_noisy_np = img_np +np.random.normal(scale=sigma, size=img_np.shape)
            _,mask  = get_bernoulli_mask(img_pil,0.5)
            img_noisy_np = np.clip(img_noisy_np , 0, 1).astype(np.float32)
            img_noisy_masked  = mask * img_noisy_np            
            # img_noisy_pil = np_to_pil(img_noisy_np)
            # img_noisy_pil.save(os.path.join(train_noisy_folder, filename + '.png'))
        
            break  # exit the loop

    noisy_psnr = compare_psnr(img_np,img_noisy_masked)
    print("noisy psnr:",noisy_psnr)
    noisy_psnr_list.append(noisy_psnr)
    print(f'Starting vanilla DIP on {ino} using {optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta})')
    print(f"Noisy PSNR is '{noisy_psnr}'")
    k=5
    # Modify input and output depths
    output_depth = 3
    num_channels = [128] *k
    print(num_channels)

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
    print (s2)

        
        
    # macs, params = get_model_complexity_info(net, (1,32, 512, 512), as_strings=True,
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # flops, macs, params = calculate_flops(model=net, 
    #                                   input_shape=(1,3,512,512),
    #                                   output_as_string=True,
    #                                   output_precision=4)
    # print("dense UnetFLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))
    # net = skip(num_input_channels=input_depth,
    #             num_output_channels=3,
    #             num_channels_down=[128] * 5,
    #             num_channels_up=[128] * 5,
    #             num_channels_skip=[4] * 5,
    #             upsample_mode='bilinear',
    #             downsample_mode='stride',
    #             need_sigmoid=True,
    #             need_bias=True,
    #             pad='reflection',
    #             act_fun='LeakyReLU').type(dtype)
    
    # print_nonzeros(net)

    # net = cnn( num_input_channels=input_depth, num_output_channels=output_depth,
    #    num_layers=3,
    #    need_bias=True, pad='zero',
    #    act_fun='LeakyReLU').type(dtype)
   
    
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
    

    def closure_sgd(net_input,img_var,mask_var,noise_var):
        img_var = np_to_torch(img_var).type(dtype)
        noise_var = np_to_torch(noise_var).type(dtype)
        mask_var = np_to_torch(mask_var).type(dtype)
        out = net(net_input)
        total_loss = mse(out*mask_var, noise_var*mask_var)     
        optimizer.zero_grad()     
        total_loss.backward()
        optimizer.step()               
        out_np = out.detach().cpu().numpy()
        img_np = img_var.detach().cpu().numpy()
        psnr_gt  = compare_psnr(img_np, out_np)             
        out_np = out.detach().cpu().numpy()
        img_np = img_var.detach().cpu().numpy()
        psnr_gt  = compare_psnr(img_np, out_np)

        return psnr_gt,out_np

    fileid = f'{optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta},reg={reg})'
    #outdir = f'data/denoising/Dataset/mask/{ino}/{fileid}'
    outdir = f'data/denoising_inpaint/Set14/mask/{ino}/decoder/{sigma}'
    os.makedirs(f'{outdir}', exist_ok=True)
    
    for j in range(max_steps):
        #optimizer.zero_grad()
        psnr,out = closure_sgd( net_input, img_np, mask, img_noisy_masked)

        if j%show_every==0 and j!=0:
            #print(psnr_lists[0][-1],psnr_lists[1][-1],psnr_lists[-1][-1]) 
            #print(psnr_lists[0][-1])   
            #e1,e2= get_hessian_eigenvalues(net, ind_loss, net_input_list, img_np_list, img_noisy_np_list,neigs=2)
            #jac = get_jac_norm(net,net_input_list)
            #trace = get_trace(net, ind_loss, net_input_list, img_np_list, img_noisy_np_list)
            #print(e1,jac,trace)
            #print(e1,jac,trace)
            #print(f"At step '{j}', e1 is '{e1}', psnr is '{psnr}'")

            print(f"At step '{j}', psnr is '{psnr}'")
            psnr_list.append(psnr)  
            # out = (out - out.min()) / (out.max() - out.min())   
            # plt.imsave(f'{outdir}/out_images/out_image_{ino}_{j}.png', out, cmap='gray')
            #sharp.append(e1)
            #jacl.append(jac)
            #tr.append(trace)
    # end_time = time.time()
    # duration = end_time - start_time 
    # print("duration is:",duration)                
        #if j%10000==0:
            #eig,weight = get_hessian_spectrum(net, ind_loss, net_input_list, img_np_list, img_noisy_np_list,iter= 100, n_v=1)
    ##plot and save psnr list in train folder with figure name including ino 
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
    parser.add_argument("--ino", type=int, default=0, help="image index ")
    args = parser.parse_args()
    
    main(images=args.images, lr=args.lr, max_steps=args.max_steps, optim= args.optim,reg=args.reg,sigma = args.sigma, num_layers = args.num_layers, show_every = args.show_every, beta = args.beta, device_id = args.device_id,ino = args.ino, weight_decay = args.decay)
        
    
    
        
        
    
    

