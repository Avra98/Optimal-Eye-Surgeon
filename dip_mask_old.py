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
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sam import SAM

import argparse

def inverse_sigmoid(x):
    return torch.log(torch.tensor(x) / torch.tensor(1 - x))

def main(images: list, lr: float, max_steps: int, optim: str, reg: float = 0.0, sigma: float = 0.2,
          num_layers: int = 4, show_every: int=1000, device_id: int = 0,beta: float = 0.0,
          ino : int =0,weight_decay: float = 0.0, mask_opt: str = "det", noise_steps: int = 80000,kl: float = 1e-5,sparsity: float = 0.05):
    sparsity=args.sparsity
    torch.cuda.set_device(device_id)
    torch.cuda.current_device() 
    prior_sigma=inverse_sigmoid(args.sparsity)

    def normalize_image(img):
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val)  

    # def compare_psnr(img1, img2):
    #     MSE = np.mean(np.abs(img1-img2)**2)
    #     psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
    #     return psnr 

    
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
            #print(ino)

            img_noisy_np = img_np +np.random.normal(scale=sigma, size=img_np.shape)
            img_noisy_np = np.clip(img_noisy_np , 0, 1).astype(np.float32)

            img_np_list.append(img_np)
            img_noisy_np_list.append(img_noisy_np)
            
            img_noisy_pil = np_to_pil(img_noisy_np)
            img_noisy_pil.save(os.path.join(train_noisy_folder, filename + '.png'))

            break  # exit the loop


    # ## please delete after experiment
    # img_shape = (3, 512, 512)  # (channels, height, width)
    # square_size = 8
    # chessboard_image = generate_specific_quarter_chessboard(img_shape, square_size, noise_level=0.0, quarter=1)
    # img_np = chessboard_image
    
    noisy_psnr = compare_psnr(img_np,img_noisy_np)
    noisy_psnr_list.append(noisy_psnr)
    print(f"Noisy PSNR is '{noisy_psnr}'")
    

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
    print(img_np.shape,img_noisy_np.shape,net_input.shape)
    net = skip(
        input_depth, output_depth,
        num_channels_down = [16, 32, 64, 128, 128, 128,128, 128, 128, 128][:num_layers],
        num_channels_up   = [16, 32, 64, 128, 128, 128,128, 128, 128, 128][:num_layers],
        # num_channels_down = [32, 64, 128, 256, 256, 256,256, 256, 256, 256][:num_layers],
        # num_channels_up   = [32, 64, 128, 256, 256, 256,256, 256, 256, 256][:num_layers],
        # num_channels_down = [64, 128, 256, 512, 512, 512,512, 512, 512, 512][:num_layers],
        # num_channels_up   = [64, 128, 256, 512, 512, 512,512, 512, 512, 512][:num_layers],
        # num_channels_down = [128, 256, 512, 1024, 1024, 1024,1024, 1024, 1024, 1024][:num_layers],
        # num_channels_up   = [128, 256, 512, 1024, 1024, 1024,1024, 1024, 1024, 1024][:num_layers],
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
    
    ## print number of network parameters
    # s  = sum(np.prod(list(p.size())) for p in net.parameters())
    # print ('Number of params: %d' % s)
    ## torch load existing model from path 
    # state_dict = torch.load(f'data/denoising/Set14/mask/{ino}/vanilla/0.1/model_early{ino}.pth')    
    # net.load_state_dict(state_dict)
    ## apply normal initialization 
    #net.apply(init_weights)
    ## multiply all the initial weights by 5
    #net.apply(lambda m: weights_init(m, scale=0.1))
    # for param in net.parameters():
    #     print(param)

    
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

    
    # print(f"Starting optimization with optimizer '{optim}'")
    # if optim =="SGD":
    #     optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay = weight_decay,momentum = beta)
    # elif optim =="ADAM":
    #     optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay = weight_decay)
    # elif optim =="SAM":
    #     optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay = weight_decay,momentum = beta)
        # base_opt = torch.optim.SGD
        # optimizer = SAM(net.parameters(), base_opt, rho=args.reg, adaptive=False, lr=args.lr, weight_decay = weight_decay,momentum = beta) 
          

    #outdir = f'data/denoising/Dataset/mask/{ino}/unet/{mask_opt}/{prior_sigma}/{kl}'
    outdir = f'data/denoising/Set14/mask/{ino}/sparsity/{mask_opt}/{sparsity}/{kl}/{num_layers}layers/l1_cent_smalllrs'
    print(f"Output directory: {outdir}")
    os.makedirs(f'{outdir}/out_images/', exist_ok=True)

    m=0
    p,quant_loss = learn_quantization_probabilities_dip(net,net_input, img_np, img_noisy_np, num_steps,
                                                         lr,ino, q=2,kl=args.kl,prior_sigma=prior_sigma,sparsity=sparsity)
    # p,quant_loss = learn_quantization_probabilities_dip(net,net_input, img_np, img_np, num_steps, lr,ino, q=2,kl=args.kl,prior_sigma=args.prior_sigma)
            ## save p in the outdir foldr     
    mask = make_mask_with_sparsity(p, sparsity) 
    masked_model = mask_network(mask,net)  
    # save masked model and net_input_list in the outdir folder
    with open(f'{outdir}/masked_model_{ino}.pkl', 'wb') as f:
        cPickle.dump(masked_model, f)
    with open(f'{outdir}/net_input_list_{ino}.pkl', 'wb') as f:
        cPickle.dump(net_input, f)
    with open(f'{outdir}/mask_{ino}.pkl', 'wb') as f:
       cPickle.dump(mask, f)
    ## save p
    with open(f'{outdir}/p_{ino}.pkl', 'wb') as f:
        cPickle.dump(p, f)     
    #     
                
    with torch.no_grad():
        if mask_opt=='single':          
            out = draw_one_mask(p, net,net_input)
        elif mask_opt=='multiple':
            out = draw_multiple_masks(p, net,net_input)
        else:
            out = deterministic_rounding(p, net,net_input,sparsity=args.sparsity)


        # out_np = out.detach().cpu().numpy()[0]
        # img_np = img_var.detach().cpu().numpy()[0]
        out_np = torch_to_np(out)
        img_var = np_to_torch(img_np)
        img_np = torch_to_np(img_var)

        print(out_np.shape, img_np.shape, img_noisy_np.shape)
        psnr_gt  = compare_psnr(img_np, out_np)
        
        print("PSNR of output image is: ", psnr_gt)        

        output_paths = [
        f"{outdir}/out_images/out_{ino}.png",
        f"{outdir}/out_images/img_np_{ino}.png",
        f"{outdir}/out_images/img_noisy_np_{ino}.png"]
            
        
        images_to_save = [out_np.transpose(1,2,0), img_np.transpose(1,2,0), img_noisy_np.transpose(1,2,0)]
        for path, img in zip(output_paths, images_to_save):
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()
        # plt.hist(p.cpu().numpy(), bins=50, alpha=0.5, label='Logits')

        # plt.title(f'Logits Histogram')
        # plt.xlabel('Logit (p) Values')
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
                plt.savefig(f"{outdir}/psnr_{ino}.png") 

    #plot_psnr(psnr_lists)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")
    
    parser.add_argument("--images", type=str, default = ["Lena512rgb"], help="which image to denoise")
    parser.add_argument("--lr", type=float,  default=1e-2, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=60000, help="the maximum number of gradient steps to train for")
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
    parser.add_argument("--noise_steps", type=int, default=60000, help="numvere of steps for noise")
    parser.add_argument("--kl", type=float, default=1e-9, help="regularization strength of kl")
    #parser.add_argument("--prior_sigma", type=float, default=0.0, help="prior mean")
    parser.add_argument("--sparsity", type=float, default=0.05, help="fraction to keep")
    args = parser.parse_args()
    
    main(images=args.images, lr=args.lr, max_steps=args.max_steps, 
         optim= args.optim,reg=args.reg,sigma = args.sigma, num_layers = args.num_layers, 
         show_every = args.show_every, beta = args.beta, device_id = args.device_id,ino = args.ino,
         weight_decay = args.decay,mask_opt = args.mask_opt, noise_steps = args.noise_steps,kl = args.kl,sparsity=args.sparsity)
        
    
    
        