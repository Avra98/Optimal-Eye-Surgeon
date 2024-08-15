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
from utils.pruning import *

# Suppress warnings
# warnings.filterwarnings("ignore")

# Enable CUDA
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def main(image_name: str, max_steps: int, sigma: float = 0.2,
         show_every: int = 1000, device: str = 'cuda:0', 
         sparsity: float = 0.95, mask_type: str = 'structured', force: bool = False):

    basedir = f'sparse_models/{image_name}/sparse-{sparsity}/noise-{sigma}'
    outdir = f'{basedir}/out_sparsenet'

    if not os.path.exists(basedir):
        print(f"Model does not exist for {image_name}/sparse-{sparsity}/noise-{sigma}")
        return

    # if there are already results here, quit and don't overwrite
    if os.path.exists(outdir):
        if not force:
            print(f"Results already exist for {image_name}/sparse-{sparsity}/noise-{sigma}")
            print('If you want to run the experiment again, delete the existing results first or allow overwrite with --force')
            return
        else:
            print(f"WARNING: You may potentially overwrite results for {image_name}/sparse-{sparsity}/noise-{sigma}")

    os.makedirs(outdir, exist_ok=True)

    logger = get_logger(
        LOG_FORMAT='%(asctime)s %(module)s %(levelname)-8s %(message)s', 
        LOG_NAME='main', 
        LOG_FILE_INFO=f'{outdir}/info.txt', LOG_FILE_DEBUG=f'{outdir}/debug.txt')

    device = f'cuda:{device}' if device.isdigit() else device
    torch.set_default_device(device)
    torch.get_default_device()

    train_folder = 'images'
    img_np, img_noisy_np, noisy_psnr = load_image(train_folder, image_name, sigma)

    logger.info(f"Target sparsity {sparsity*100}% on image '{image_name}' with sigma={sigma}")
    logger.info(f"Outdir: {outdir}")

    with open(f'{basedir}/net_init.pkl', 'rb') as f:
        net_init = cPickle.load(f)
    with open(f'{basedir}/net_input.pkl', 'rb') as f:
        net_input = cPickle.load(f)
    # with open(f'{basedir}/mask_{image_name}.pkl', 'rb') as f:
    #     mask = cPickle.load(f)
    with open(f'{basedir}/p-star.pkl', 'rb') as f:
        p = cPickle.load(f)
    
    # # print out all the module names that are actually modules, not containers
    # for name, module in net.named_modules():
    #     if len(list(module.children())) == 0:
    #         print(module)
    
    # exit()

    p_net = copy.deepcopy(net_init)
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

    if mask_type == 'structured':
        mask = make_mask_torch_pruneln(p_net, sparsity=sparsity)
    elif mask_type == 'unstructured':
        # unstructured masking
        mask = make_mask_unstructured(p, sparsity=sparsity)
    else: 
        raise ValueError(f"Mask type '{mask_type}' not supported")

    logger.info('Actual sparsity achived: %s', torch.sum(mask == 0).item() / mask.size(0))
    mask_network(mask, net_init)

    logger.info("=== START SPARSE TRAINING ===")
    ssim, psnr, out = train_sparse(net_init, net_input, mask, img_np, img_noisy_np,
                             max_step=max_steps, show_every=show_every, device=device)
    np.savez(f'{outdir}/psnr.npz', psnr=psnr)
    np.savez(f'{outdir}/ssim.npz', ssim=ssim)

    with torch.no_grad():
        add_hook_feature_maps(net_init)

        out_np = net_init(net_input).detach().cpu().numpy()
        plot_feature_maps(f'{outdir}/fm_{mask_type}.png', net_init.feature_maps)

        img_var = np_to_torch(img_np)
        img_np = img_var.detach().cpu().numpy()
        psnr_gt = compare_psnr(img_np, out_np)
        logger.info("PSNR of output image is: %s", psnr_gt)
        logger.info("SSIM of output image is: %s", structural_similarity(img_np[0], out_np[0], 
                                                                 channel_axis=0, data_range=img_np.max() - img_np.min()))

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
        plt.savefig(f'{outdir}/psnr.png')
        plt.close()

        plt.plot(ssim)
        plt.title('SSIM vs Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('SSIM')
        plt.savefig(f'{outdir}/ssim.png')
        plt.close()

    torch.cuda.empty_cache()
    logger.info("Experiment done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using sparse DIP")

    image_choices = [
        'baboon', 'barbara', 'lena', 'pepper'
    ]

    parser.add_argument("--image_name", type=str, choices=image_choices, default='pepper', help="which image to denoise")
    parser.add_argument("--sparsity", type=float,  default=0.95, help="which image to denoise")
    parser.add_argument("--max_steps", type=int, default=60000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show every N steps")
    parser.add_argument("--mask_type", type=str, default='structured', help="mask type")
    parser.add_argument("--device", type=str, default='cuda:0', help="specify which GPU")
    parser.add_argument('-f', "--force", action='store_true', default=False, help="overwrite existing results?")

    args = parser.parse_args()

    main(image_name=args.image_name, sparsity=args.sparsity, max_steps=args.max_steps, sigma=args.sigma,
         show_every=args.show_every, mask_type=args.mask_type, device=args.device, force=args.force)
