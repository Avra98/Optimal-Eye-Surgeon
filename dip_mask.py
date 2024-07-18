from __future__ import print_function
import matplotlib.pyplot as plt
import os
import warnings
import torch
import torch.optim
import argparse
import yaml
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from utils.denoising_utils import *
from utils.quant import *
from utils.imp import *
from models import *

# Suppress warnings
warnings.filterwarnings("ignore")

# Enable CUDA
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

dtype = torch.cuda.FloatTensor
torch.set_default_tensor_type(dtype)

def main(image_name: str, lr: float, max_steps: int,
         sigma: float = 0.2, num_layers: int = 4, show_every: int = 1000, device_id: int = 0,
         mask_opt: str = "det", kl: float = 1e-9, sparsity: float = 0.05):

    ### === Setup ===
    torch.cuda.set_device(device_id)
    torch.cuda.current_device()
    prior_sigma = inverse_sigmoid(sparsity)

    train_folder = 'images'
    img_np, img_noisy_np, noisy_psnr = load_image(train_folder, image_name, sigma)

    input_depth = 32
    output_depth = 3
    num_steps = max_steps

    net_input = get_noise(input_depth, "noise", img_np.shape[1:]).type(dtype)

    net = skip(
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
    ).type(dtype)

    outdir = f'sparse_models/{image_name}'
    os.makedirs(f'{outdir}/out_images/', exist_ok=True)

    print(f"Now mask with sparsity level '{sparsity}' is starting to get learned on image '{image_name}' with sigma={sigma}.")
    print(f"The noisy PSNR is '{noisy_psnr}'.")
    print(f"All results will be saved in: {outdir}")

    ### === OES ===

    # Learn quantization probability, p, corresponding to each parameter
    p, quant_loss = learn_quantization_probabilities_dip(
        net, net_input, img_np, img_noisy_np, num_steps, lr, image_name, q=2, 
        kl=kl, prior_sigma=prior_sigma, sparsity=sparsity, show_every=show_every)

    mask = make_mask_with_sparsity(p, sparsity)
    masked_model = mask_network(mask, net)

    with open(f'{outdir}/masked_model_{image_name}.pkl', 'wb') as f:
        cPickle.dump(masked_model, f)
    with open(f'{outdir}/net_input_list_{image_name}.pkl', 'wb') as f:
        cPickle.dump(net_input, f)
    with open(f'{outdir}/mask_{image_name}.pkl', 'wb') as f:
        cPickle.dump(mask, f)
    with open(f'{outdir}/p_{image_name}.pkl', 'wb') as f:
        cPickle.dump(p, f)

    with torch.no_grad():
        if mask_opt == 'single':
            out = draw_one_mask(p, net, net_input)
        elif mask_opt == 'multiple':
            out = draw_multiple_masks(p, net, net_input)
        else:
            out = deterministic_rounding(net, net_input)

        out_np = torch_to_np(out)
        img_var = np_to_torch(img_np)
        img_np = torch_to_np(img_var)

        psnr_gt = compare_psnr(img_np, out_np)
        print(f"PSNR of output image is: {psnr_gt}")

        output_paths = [
            f"{outdir}/out_images/out_{image_name}.png",
            f"{outdir}/out_images/img_np_{image_name}.png",
            f"{outdir}/out_images/img_noisy_np_{image_name}.png"
        ]

        images_to_save = [out_np.transpose(1, 2, 0), img_np.transpose(1, 2, 0), img_noisy_np.transpose(1, 2, 0)]
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
        plt.savefig(f'{outdir}/out_images/qquant_loss_{image_name}.png')

    torch.cuda.empty_cache()
    print("Experiment done")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")

    image_choices = [
        'baboon', 'barbara', 'lena', 'pepper'
    ]

    parser.add_argument("--image_name", type=str, choices=image_choices, help="which image to denoise")
    parser.add_argument("--lr", type=float, help="the learning rate")
    parser.add_argument("--max_steps", type=int, help="the maximum number of gradient steps to train for")
    parser.add_argument("--sigma", type=float, help="noise-level")
    parser.add_argument("--num_layers", type=int, help="number of layers")
    parser.add_argument("--show_every", type=int, help="show_every")
    parser.add_argument("--device_id", type=int, help="specify which gpu")
    parser.add_argument("--mask_opt", type=str, help="mask type")
    parser.add_argument("--kl", type=float, help="regularization strength of kl")
    parser.add_argument("--sparsity", type=float, help="fraction to keep")
    parser.add_argument("-f", "--file", type=str, default='configs/config_mask.yaml', help="YAML configuration file, options passed on the command line override these")

    args = parser.parse_args()

    default_config = {
        'image_name': 'pepper',
        'lr': 0.01,
        'max_steps': 60000,
        'sigma': 0.1,
        'num_layers': 6,
        'show_every': 1000,
        'device_id': 0,
        'mask_opt': 'det',
        'kl': 1e-9,
        'sparsity': 0.05
    }

    config = set_config(args.file, default_config)

    main(
        image_name=config.get('image_name', default_config['image_name']),
        lr=config.get('lr', default_config['lr']),
        max_steps=config.get('max_steps', default_config['max_steps']),
        sigma=config.get('sigma', default_config['sigma']),
        num_layers=config.get('num_layers', default_config['num_layers']),
        show_every=config.get('show_every', default_config['show_every']),
        device_id=config.get('device_id', default_config['device_id']),
        mask_opt=config.get('mask_opt', default_config['mask_opt']),
        kl=config.get('kl', default_config['kl']),
        sparsity=config.get('sparsity', default_config['sparsity'])
    )