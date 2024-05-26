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
from utils.sharpness import *
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

def main(images: list, lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2, num_layers: int = 4, show_every: int=1000, device_id: int = 0,beta: float = 0.0,ino : int =0,weight_decay: float = 0.0, init_scale: float = 1.0):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device() 
    seed = 42  # You can choose any seed value
    torch.manual_seed(seed)
    np.random.seed(seed)

    def normalize_image(img):
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val)  

       
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
            imsize = -1
            #img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_pil = Image.open(file_path)
            #img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_pil = resize_and_crop(img_pil, max(img_pil.size))
            img_np = pil_to_np(img_pil)

            img_noisy_np = img_np +np.random.normal(scale=sigma, size=img_np.shape)
            img_noisy_np = np.clip(img_noisy_np , 0, 1).astype(np.float32)
            img_np_list.append(img_np)
            img_noisy_np_list.append(img_noisy_np)
            
            img_noisy_pil = np_to_pil(img_noisy_np)
            img_noisy_pil.save(os.path.join(train_noisy_folder, filename + '.png'))
        
            break  # exit the loop



    # def np_to_pil(img_np):
    #     """Convert a NumPy array to a PIL Image."""
    #     return Image.fromarray(np.uint8(img_np * 255))

    # def create_constant_image_with_noise(size, constant_value, sigma):
    #     """
    #     Create a constant image of a given size and constant value, then add Gaussian noise.

    #     :param size: Tuple of (width, height) for the image size.
    #     :param constant_value: The constant value for all pixels in the image.
    #     :param sigma: Standard deviation of Gaussian noise to add.
    #     :return: A noisy PIL image.
    #     """
    #     # Create a constant image
    #     img_np = np.full((3,size[1], size[0]), constant_value, dtype=np.float32)

    #     # Add Gaussian noise
    #     img_noisy_np = img_np + np.random.normal(scale=sigma, size=img_np.shape)
    #     img_noisy_np = np.clip(img_noisy_np, 0, 1).astype(np.float32)


    #     return img_np,img_noisy_np

    # # Example usage
    # size = (512, 512)  # Image size (width, height)
    # constant_value = 0.5  # Constant pixel value (between 0 and 1 for grayscale)
    # sigma = 0.05  # Noise standard deviation

    # img_np,img_noisy_np = create_constant_image_with_noise(size, constant_value, sigma) 
        
    def create_low_freq_image_with_noise(size, patch_size, sigma):
        """
        Create an image with low-frequency patches and add Gaussian noise.

        :param size: Tuple of (width, height) for the image size.
        :param patch_size: Size of the square patches (width, height).
        :param sigma: Standard deviation of Gaussian noise to add.
        :return: Tuple of original and noisy image as NumPy arrays.
        """
        # Initialize an empty image
        img_np = np.zeros((3, size[1], size[0]), dtype=np.float32)
        
        # Create low-frequency patches by filling certain areas with constant values
        for y in range(0, size[1], patch_size[1]):
            for x in range(0, size[0], patch_size[0]):
                # Define the patch value as a random float between 0 and 1
                patch_value = np.random.rand()
                img_np[:, y:y+patch_size[1], x:x+patch_size[0]] = patch_value
        
        # Normalize the image to have values between 0 and 1
        img_np = img_np / img_np.max()
        
        # Add Gaussian noise
        img_noisy_np = img_np + np.random.normal(scale=sigma, size=img_np.shape)
        img_noisy_np = np.clip(img_noisy_np, 0, 1).astype(np.float32)

        return img_np, img_noisy_np

    # Example usage
    size = (512, 512)  # Image size (width, height)
    patch_size = (256, 256)  # Size of low-frequency patches
    sigma = 0.05  # Noise standard deviation

    img_np, img_noisy_np = create_low_freq_image_with_noise(size, patch_size, sigma)
    ## print and image save both the original and noisy image
    img_pil = np_to_pil(img_np)
    img_noisy_pil = np_to_pil(img_noisy_np)
    img_pil.save('data/denoising/eos/original.png')
    img_noisy_pil.save('data/denoising/eos/noisy.png')


    print(img_np.shape)

    noisy_psnr = compare_psnr(img_np,img_noisy_np)
    print("noisy psnr:",noisy_psnr)
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
        
    net_input= get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype) 
    print("input dim:", net_input.shape)# [1, 3, 256, 256]
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
    
    ##scale the network parameters by 0.01 
    for param in net.parameters():
        param.data = param.data*args.init_scale


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
    

    def closure_sgd(net_input,img_var,noise_var):

        img_var = np_to_torch(img_var).type(dtype)
        noise_var = np_to_torch(noise_var).type(dtype)
        #start_time = time.time()  
        out = net(net_input)

        # with torch.no_grad():
        #     print(net_input.shape,img_var.shape,out.shape)
        total_loss = mse(out, noise_var)


        # if optim=="SGD" or  optim=="ADAM":      
        optimizer.zero_grad() 
          
        total_loss.backward()
        # end_time = time.time()
        # duration = end_time - start_time
        # print("duration is:",duration) 

        optimizer.step()
       
        # end_time = time.time()
        # duration = end_time - start_time
        #print("duration is:",duration)        
        # elif optim=="SAM":
        #     total_loss.backward() 
        #     optimizer.first_step(zero_grad=True)
        #     mse(net(net_input), noise_var).backward()
        #     optimizer.second_step(zero_grad=True)                
        out_np = out.detach().cpu().numpy()
        img_np = img_var.detach().cpu().numpy()
        psnr_gt  = compare_psnr(img_np, out_np)

        return psnr_gt,out_np 

    fileid = f'{optim}(sigma={sigma},lr={lr},decay={weight_decay},beta={beta},reg={reg})'
    #outdir = f'data/denoising/Dataset/mask/{ino}/{fileid}'
    outdir = f'data/denoising/eos/{args.init_scale}/{ino}/{optim}'
    os.makedirs(f'{outdir}', exist_ok=True)
    previous_model_state = None
    for j in range(max_steps):
        #optimizer.zero_grad()
        psnr,out = closure_sgd( net_input, img_np, img_noisy_np)
        # if j ==2000:
        #     ## save the model
        #     torch.save(net.state_dict(), f'{outdir}/model_early{ino}.pth')
            
        if j == show_every - 1:
            # Save the model weights of convolutional layers at show_every - 1
            previous_model_state = {name: param.clone() for name, param in net.named_parameters() if param.dim() == 4}

        if j%show_every==0 and j!=0:
            e1,e2= get_hessian_eig(net, ind_loss, net_input, img_np, img_noisy_np,neigs=2)
            current_model_state = {name: param for name, param in net.named_parameters() if param.dim() == 4}          
            ## save the output image of the network with j in the name of the imahe 
            out_img = np_to_pil(out[0,:,:,:])
            out_img.save(f'{outdir}/out_{ino}_{j}.png')

            # # Calculate the norm difference for each layer
            # for name, param in current_model_state.items():
            #     prev_param = previous_model_state[name]
            #     num_elements = param.numel()  # Get the number of elements in the parameter tensor
                
            #     # Calculate the L2 norm of the difference and divide by the number of elements
            #     diff = (param - prev_param).norm().item() / num_elements
                
            #     # Calculate the L2 norm of the current parameters and divide by the number of elements
            #     current_norm = param.norm().item() / num_elements
                
            #     print(f"Layer {name}, iter:{j}, Weight Difference: {diff}, Current Norm: {current_norm}")

            # # Update the previous_model_state for the next interval
            # previous_model_state = current_model_state.copy()

            print(f"At step '{j}', e1 is '{e1}', psnr is '{psnr}'")
            #print(f"At step '{j}', psnr is '{psnr}'")
            psnr_list.append(psnr)  
            sharp.append(e1)



    # end_time = time.time()
    # duration = end_time - start_time 
    # print("duration is:",duration)                
        #if j%10000==0:
            #eig,weight = get_hessian_spectrum(net, ind_loss, net_input_list, img_np_list, img_noisy_np_list,iter= 100, n_v=1)
    ##plot and save psnr list in train folder with figure name including ino 
    plt.plot(psnr_list)
    plt.savefig(f'{outdir}/psnr_{ino}.png')
    plt.close()
    plt.plot(sharp)
    plt.savefig(f'{outdir}/sharp_{ino}.png')
    plt.close()
    ## save the list "psnr" as an npz file and save in the outdir folder
    np.savez(f'{outdir}/psnr_{ino}.npz', psnr=psnr_list)
    np.savez(f'{outdir}/sharp_{ino}.npz', sharp=sharp)  
    

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
    parser.add_argument("--init_scale", type=float, default=1.0, help="initial scale")
    args = parser.parse_args()
    
    main(images=args.images, lr=args.lr, max_steps=args.max_steps, optim= args.optim,reg=args.reg,sigma = args.sigma, num_layers = args.num_layers,
          show_every = args.show_every, beta = args.beta, device_id = args.device_id,
          ino = args.ino, weight_decay = args.decay, init_scale = args.init_scale)
        
    
    
        
        
    
    

