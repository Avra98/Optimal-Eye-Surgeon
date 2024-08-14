from __future__ import print_function
import matplotlib.pyplot as plt
import os
import warnings
import torch
import torch.optim
import argparse
import logging
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *
from utils.quant import *
from utils.imp import *
from utils.pruning import * 
from models import *

# Suppress warnings
# warnings.filterwarnings("ignore")

# Enable cuDNN benchmark for performance
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# # TODO: finish implementing logging

def main(image_name: str, lr: float, max_steps: int,
         sigma: float = 0.2, num_layers: int = 4, show_every: int = 1000, device: str = 'cuda:0',
         mask_opt: str = "det", kl: float = 1e-9, sparsity: float = 0.05, force: bool = False):

    basedir = f'sparse_models/{image_name}/sparse-{sparsity}/noise-{sigma}'
    # if there are already results here, quit and don't overwrite
    if os.path.exists(basedir):
        if not force:
            print(f"Results already exist for {image_name} with sparse-{sparsity} and noise-{sigma}")
            print('If you want to run the experiment again, delete the existing results first or overwrite with --force')
            return
        else:
            print(f"WARNING: Overwriting results for {image_name} with sparse-{sparsity} and noise-{sigma}")

    outdir = f'{basedir}/out_mask'
    os.makedirs(outdir, exist_ok=True)

    logger = get_logger(
        LOG_FORMAT='%(asctime)s %(moduels)s %(levelname)-8s %(message)s', 
        LOG_NAME='sparse', 
        LOG_FILE_INFO=f'{outdir}/info.txt', LOG_FILE_DEBUG=f'{outdir}/debug.txt')

    device = f'cuda:{device}' if device.isdigit() else device
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)

    logger.info(
        f'DEVICE: {device}\n'
        f'LR: {lr}\n'
        f'OPTIM: ADAM\n'
        f'SPARSITY: {sparsity}\n'
        f'NOISE: {sigma}\n'
        f'KL REG: {kl}\n'
        f'LAYERS: {num_layers}\n'
        f'...Showing every {show_every} steps for {max_steps} iterations...'
    )

    prior_sigma = inverse_sigmoid(sparsity)

    train_folder = 'images'
    img_np, img_noisy_np, noisy_psnr = load_image(train_folder, image_name, sigma)
    noisy_ssim = structural_similarity(img_np[0], img_noisy_np[0], 
                                       channel_axis=0, data_range=img_noisy_np.max() - img_noisy_np.min())

    input_depth = 32
    output_depth = 3

    net_input = get_noise(input_depth, "noise", img_np.shape[1:])

    net = UNetCustom(
        input_depth, output_depth,
        num_channels_down=[16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_up=[16, 32, 64, 128, 128, 128][:num_layers],
        upsample_mode='nearest',
        downsample_mode='avg',
        need1x1_up=False,
        filter_size_down=5,
        filter_size_up=3,
        need_sigmoid=True,
        need_bias=True,
        pad='reflection',
        act_fun='LeakyReLU'
    )

    with open(f'{basedir}/net_init.pkl', 'wb') as f:
        cPickle.dump(net, f)
    with open(f'{basedir}/net_input.pkl', 'wb') as f:
        cPickle.dump(net_input, f)

    logger.info(f"Saved initial network and input to {basedir}")


    ### === Mask learning algorithm === ###

    logger.info(f"Noisy PSNR: '{noisy_psnr}', noisy SSIM: '{noisy_ssim}'")

    # Learn quantization probability, p, corresponding to each parameter
    p, quant_loss, p_sig = learn_quantization_probabilities_dip(
        net, net_input, img_np, img_noisy_np, num_steps=max_steps, lr=lr, q=2, 
        kl=kl, prior_sigma=prior_sigma, sparsity=sparsity, show_every=show_every)

    unstructured_mask = make_mask_unstructured(p, sparsity)
    structured_mask = make_mask_torch_pruneln(p, sparsity)

    _ = copy.deepcopy(net)
    
    net_unst_mask = mask_network(unstructured_mask, net)
    net_st_mask = mask_network(structured_mask, _)

    # with open(f'{basedir}/mask_unstructured_{image_name}.pkl', 'wb') as f:
    #     cPickle.dump(unstructured_mask, f)
    # with open(f'{basedir}/mask_structured_{image_name}.pkl', 'wb') as f:
    #     cPickle.dump(structured_mask, f)
    with open(f'{basedir}/p-star.pkl', 'wb') as f:
        cPickle.dump(p, f)


    # TODO plot out feature maps after each layer: https://cs231n.github.io/convolutional-networks/
    # TODO also do for train_sparse
    with torch.no_grad():
        add_hook_feature_maps(net_unst_mask)
        add_hook_feature_maps(net_st_mask)
        if mask_opt == 'single':
            out_structured = draw_one_mask(p, net_unst_mask, net_input)
            out_unstructured = draw_one_mask(p, net_st_mask, net_input)
        elif mask_opt == 'multiple':
            out_structured = draw_multiple_masks(p, net_unst_mask, net_input)
            out_unstructured = draw_multiple_masks(p, net_st_mask, net_input)
        else:
            out_structured = net_unst_mask(net_input)
            out_unstructured = net_st_mask(net_input)
        
        plot_feature_maps(f'{outdir}/fm_unstructured', net_unst_mask.feature_maps)
        plot_feature_maps(f'{outdir}/fm_torch_prune', net_st_mask.feature_maps)

        out_np_unstruct = torch_to_np(out_structured)
        out_np_struct = torch_to_np(out_unstructured)
        img_var = np_to_torch(img_np)
        img_np = torch_to_np(img_var)

        # TODO time and compare training speeds for structured and unstructured masks
        psnr_gt_unstrct = compare_psnr(img_np, out_np_unstruct)
        ssim_gt_unstrct = structural_similarity(img_np[0], out_np_unstruct[0],
                                        channel_axis=0, data_range=out_np_unstruct.max() - out_np_unstruct.min())
        psnr_gt_strct = compare_psnr(img_np, out_np_struct)
        ssim_gt_strct = structural_similarity(img_np[0], out_np_struct[0],
                                        channel_axis=0, data_range=out_np_struct.max() - out_np_struct.min())
        logger.info("UNSTRUCTURED output PSNR: %s", psnr_gt_unstrct)
        logger.info("UNSTRUCTURED output SSIM: %s", ssim_gt_unstrct)
        logger.info("STRUCTURED output PSNR: %s", psnr_gt_strct)
        logger.info("STRUCTURED output SSIM: %s", ssim_gt_strct)


        output_paths = [
            f"{outdir}/out_unstructured.png",
            f"{outdir}/out_structured.png",
            f"{outdir}/gt.png",
            f"{outdir}/noisy.png"
        ]

        images_to_save = [
            out_np_unstruct.transpose(1, 2, 0), 
            out_np_struct.transpose(1, 2, 0),
            img_np.transpose(1, 2, 0), 
            img_noisy_np.transpose(1, 2, 0)
        ]

        for path, img in zip(output_paths, images_to_save):
            plt.imshow(img)
            plt.axis('off')
            plt.savefig(path, bbox_inches='tight', pad_inches=0)
            plt.close()

        plt.plot(range(0, len(quant_loss) * 1000, 1000), quant_loss, marker='o', linestyle='-')
        plt.title('Quantization Loss Over Training Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Quantization Loss')
        plt.grid(True)
        plt.savefig(f'{outdir}/quant_loss.png')
        
    # Plot a histogram for all the quantized weights
    plt.hist(p_sig, bins=50, alpha=0.5, label='All Layers')
    plt.title(f'Distribution of p for sparsity level {sparsity}')
    plt.xlabel('Value of p')
    plt.ylabel('Frequency')
    os.makedirs(f'sparse_models/{image_name}/histograms/', exist_ok=True)
    plt.savefig(f'{outdir}/p_histogram_kl-{kl}.png')
    plt.clf()

    torch.cuda.empty_cache()
    logger.info("Experiment done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")

    image_choices = [
        'baboon', 'barbara', 'lena', 'pepper'
    ]

    parser.add_argument("--image_name", type=str, choices=image_choices, default='pepper', required=False, help="which image to denoise")
    parser.add_argument("--lr", type=float, default=1e-2, help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=50000, help="the maximum number of gradient steps to train for")
    parser.add_argument("--optim", type=str, default="SAM", help="which optimizer")
    parser.add_argument("--reg", type=float, default=0.05, help="if regularization strength of igr")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise-level")
    parser.add_argument("--num_layers", type=int, default=6, help="number of layers")
    parser.add_argument("--show_every", type=int, default=1000, help="show_every")
    parser.add_argument("--device", type=str, default='cuda:0', help="specify which gpu")
    parser.add_argument("--beta", type=float, default=0, help="momentum for sgd")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--mask_opt", type=str, default="det", help="mask type")
    parser.add_argument("--noise_steps", type=int, default=60000, help="number of steps for noise")
    parser.add_argument("--kl", type=float, default=1e-9, help="regularization strength of kl")
    parser.add_argument("--sparsity", type=float, default=0.05, help="fraction to keep")
    parser.add_argument("--force", type=bool, default=False, help="overwrite existing results?")

    args = parser.parse_args()

    main(image_name=args.image_name, lr=args.lr, max_steps=args.max_steps, sigma=args.sigma,
         num_layers=args.num_layers, show_every=args.show_every, device=args.device,
         mask_opt=args.mask_opt, kl=args.kl, sparsity=args.sparsity, force=args.force)
