from __future__ import print_function
import matplotlib.pyplot as plt
import os
import logging
import warnings
import numpy as np
import torch
import torch.optim
import torch.nn.utils.prune as prune
import argparse
import pickle as cPickle
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity 
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from utils.denoising_utils import *
from models import *
from utils.quant import *
from utils.imp import *

# Suppress warnings
# warnings.filterwarnings("ignore")

# Enable CUDA
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def main(image_name: str, max_steps: int, sigma: float = 0.2,
         num_layers: int = 4, show_every: int = 1000, device: str = 'cuda:0', 
         ino: int = 0, sparsity: float = 0.05):
    basedir = f'sparse_models/{image_name}'
    outdir = f'{basedir}/out_sparsenet/{sigma}'
    os.makedirs(outdir, exist_ok=True)
    logger = get_logger(
        LOG_FORMAT='%(asctime)s %(levelname)-8s %(message)s', 
        LOG_NAME='sparse', 
        LOG_FILE_INFO=f'{outdir}/info.txt', LOG_FILE_DEBUG=f'{outdir}/debug.txt')

    device = f'cuda:{device}' if device.isdigit() else device
    torch.set_default_device(device)
    torch.get_default_device()

    train_folder = 'images'
    img_np, img_noisy_np, noisy_psnr = load_image(train_folder, image_name, sigma)

    input_depth = 32
    output_depth = 3

    net_orig = skip(
        input_depth, output_depth,
        num_channels_down=[16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_up=[16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_skip=[0] * num_layers,
        upsample_mode='nearest',
        downsample_mode='avg',
        need1x1_up=False,
        filter_size_down=5,
        filter_size_up=3,
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad='reflection',
        act_fun='LeakyReLU'
    )

    logger.info(f"Sparsity '{sparsity}' on image '{image_name}' with sigma={sigma}.")
    logger.info(f"Outdir: {outdir}")

    with open(f'{basedir}/net_orig_{image_name}.pkl', 'rb') as f:
        net_orig = cPickle.load(f)
    with open(f'{basedir}/net_input_list_{image_name}.pkl', 'rb') as f:
        net_input_list = cPickle.load(f)
    with open(f'{basedir}/mask_{image_name}.pkl', 'rb') as f:
        mask = cPickle.load(f)
    with open(f'{basedir}/p_{image_name}.pkl', 'rb') as f:
        p = cPickle.load(f)
    
    # print out all the module names that are actually modules, not containers
    # for name, module in net_orig.named_modules():
    #     if len(list(module.children())) == 0:
    #         print(module)

    p_net = copy.deepcopy(net_orig)
    logger.debug('p shape: %s', p.shape)
    vector_to_parameters(p[0], p_net.parameters())

    # for param in p_net.parameters():
    #     param = 
    #     w0.append(param.data.view(-1).detach().clone())
    # w0 = torch.cat(w0)
    # p = nn.Parameter(inverse_sigmoid(1/q)*torch.ones([q-1, w0.size(0)]).to(device), requires_grad=True)
    # return w0, pk

    # handwritten implementation of zeroing
    # structured_mask = make_mask_structured(net_orig, p_net)

    # torch.nn.utils.prune implementation
    # logger.info('Using prune.ln_structured for masking')
    # for module in p_net.modules():
    #     if isinstance(module, torch.nn.Conv2d):
    #         # logger.debug('Module shape: %s', module.weight.shape)
    #         before = torch.sum(module.weight != 0)
    #         # logger.debug('Non-zero weights: %s', torch.sum(module.weight != 0))
    #         prune.ln_structured(module, name='weight', n=1, amount=1-sparsity, dim=1)
    #         prune.remove(module, 'weight') #  apply the mask permanently 
    #         after = torch.sum(module.weight != 0)
    #         # logger.debug('Non-zero weights: %s', torch.sum(module.weight != 0))
    #         # logger.debug('Module mask values %s', module.weight_mask)
    #         logger.debug('Pruned %s weights from %s', (before-after).item(), module)

    # mask = parameters_to_vector(p_net.parameters())
    # mask[mask != 0] = 1

    # unstructured masking
    logger.info('Using make_mask_unstructured for masking')
    mask = make_mask_unstructured(p, sparsity=sparsity)

    logger.info('sparsity of mask: %s', torch.sum(mask == 0).item() / mask.size(0))
    mask_network(mask, net_orig)

    ssim, psnr, out = train_sparse(net_orig, net_input_list, mask, img_np, img_noisy_np,
                             max_step=max_steps, show_every=show_every, device=device)

    with torch.no_grad():
        out_np = out
        img_var = np_to_torch(img_np)
        img_np = img_var.detach().cpu().numpy()
        psnr_gt = compare_psnr(img_np, out_np)
        logger.info("PSNR of output image is: %s", psnr_gt)
        logger.info("SSIM of output image is: %s", structural_similarity(img_np[0], out_np[0], 
                                                                 channel_axis=0, data_range=img_np.max() - img_np.min()))
        np.savez(f'{outdir}/psnr_{ino}.npz', psnr=psnr)
        np.savez(f'{outdir}/ssim_{ino}.npz', ssim=ssim)

        output_paths = [
            f"{outdir}/out_{image_name}.png",
            f"{outdir}/img_np_{image_name}.png",
            f"{outdir}/img_noisy_np_{image_name}.png"
        ]

        images_to_save = [out_np[0, :, :, :].transpose(1, 2, 0), img_np[0, :, :, :].transpose(1, 2, 0), img_noisy_np.transpose(1, 2, 0)]
        for path, img in zip(output_paths, images_to_save):
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()

        plt.plot(psnr)
        plt.title('PSNR vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('PSNR')
        plt.savefig(f'{outdir}/psnr_{ino}.png')
        plt.close()

        plt.plot(ssim)
        plt.title('SSIM vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('SSIM')
        plt.savefig(f'{outdir}/ssim_{ino}.png')
        plt.close()

    torch.cuda.empty_cache()
    logger.info("Experiment done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using sparse DIP")

    image_choices = [
        'baboon', 'barbara', 'lena', 'pepper'
    ]

    parser.add_argument("--image_name", type=str, choices=image_choices, default='pepper', help="which image to denoise")
    parser.add_argument("--max_steps", type=int, default=40000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show every N steps")
    parser.add_argument("--device", type=str, default='cuda:0', help="specify which GPU")

    args = parser.parse_args()

    main(image_name=args.image_name, max_steps=args.max_steps, sigma=args.sigma,
         num_layers=args.num_layers, show_every=args.show_every, device=args.device)
