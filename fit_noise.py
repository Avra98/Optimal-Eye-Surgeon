from __future__ import print_function
import argparse
import matplotlib.pyplot as plt
import os
import glob
import warnings
import numpy as np
import torch
import torch.optim
import pickle as cPickle
from scipy.ndimage import gaussian_filter
from utils.denoising_utils import *
from utils.quant import *
from utils.inpainting_utils import *
from models import *

warnings.filterwarnings("ignore")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor


def main(lr: float, max_steps: int, reg: float = 0.0, sigma: float = 0.2,
         num_layers: int = 4, show_every: int = 1000, device_id: int = 0, beta: float = 0.0,
         ino: int = 0, weight_decay: float = 0.0, mask_opt: str = "single", 
         kl: float = 1e-5, prior_sigma: float = 0.0, net_choice: str = "sparse"):

    torch.cuda.set_device(device_id)
    torch.cuda.current_device()

    def normalize_image(img):
        min_val = np.min(img)
        max_val = np.max(img)
        return (img - min_val) / (max_val - min_val)

    def compare_psnr(img1, img2):
        MSE = np.mean(np.abs(img1-img2)**2)
        psnr = 10*np.log10(np.max(np.abs(img1))**2/MSE)
        return psnr

    def generate_chessboard_noise(img_shape, square_size, noise_level):
        """
        Generate a chessboard pattern with added noise.

        Parameters:
        img_shape (tuple): Shape of the image (channels, height, width).
        square_size (int): Size of each square in the checkerboard.
        noise_level (float): Standard deviation of the Gaussian noise added to the image.

        Returns:
        np.array: Chessboard pattern with noise.
        """
        # Create the chessboard pattern
        rows, cols = img_shape[1] // square_size, img_shape[2] // square_size
        chessboard = np.kron([[1, 0] * cols, [0, 1] * cols]
                             * rows, np.ones((square_size, square_size)))
        chessboard = np.tile(
            chessboard[:img_shape[1], :img_shape[2]], (img_shape[0], 1, 1))

        # Add Gaussian noise
        noise = np.random.normal(scale=noise_level, size=img_shape)
        noisy_chessboard = chessboard + noise
        return noisy_chessboard

    def generate_noise_images(base_noise, var):
        smoothed_noise = gaussian_filter(base_noise, sigma=var)
        return smoothed_noise

    def compute_fourier_power_spectrum(image):
        # Assuming image shape is (channels, height, width)
        if image.shape[0] > 1:
            # Convert to grayscale if it's a color image
            image = np.mean(image, axis=0)

        # Apply Fourier transform
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)

        # Compute power spectrum and use logarithmic scale
        power_spectrum = np.abs(f_shift) ** 2
        # Using log1p for numerical stability
        power_spectrum_log = np.log1p(power_spectrum)

        return power_spectrum_log

    def average_power_spectrum(power_spectra):
        avg_spectrum = np.mean(power_spectra, axis=0)
        avg_spectrum_log = np.log1p(avg_spectrum)
        return avg_spectrum_log
    train_folder = 'data/denoising/Set14'

    for i, file_path in enumerate(glob.glob(os.path.join(train_folder, '*.png'))):
        if i == ino:  # we start counting from 0, so the 3rd image is at index 2
            # Get the filename (without extension) for use in messages
            filename = os.path.splitext(os.path.basename(file_path))[0]
            imsize = -1
            img_pil = crop_image(get_image(file_path, imsize)[0], d=32)
            img_np = pil_to_np(img_pil)
            print(img_np.shape)

            break  # exit the loop

    # Modify input and output depths
    input_depth = 32
    output_depth = 3
    num_steps = args.noise_steps

    # Adjust loss function

    mse = torch.nn.MSELoss().type(dtype)

    INPUT = "noise"

    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)
    net = skip(
        input_depth, output_depth,
        num_channels_down=[16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_up=[16, 32, 64, 128, 128, 128][:num_layers],
        num_channels_skip=[0]*num_layers,
        upsample_mode='nearest',
        downsample_mode='avg',
        need1x1_up=False,
        filter_size_down=5,
        filter_size_up=3,
        filter_skip_size=1,
        need_sigmoid=True,
        need_bias=True,
        pad='reflection',
        act_fun='LeakyReLU').type(dtype)

    print("Starting optimization with optimizer ADAM ")
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    fileid = f'ADAM(sigma={sigma},lr={lr},decay={weight_decay},beta={beta})'
    outdir = f'data/denoising/Set14/mask/{ino}/{
        fileid}/{mask_opt}/{prior_sigma}/{kl}'
    print(f"Output directory: {outdir}")
    if net_choice == "sparse":
        os.makedirs(f'{outdir}/noise_learn_sparse', exist_ok=True)
    elif net_choice == "dense":
        os.makedirs(f'{outdir}/noise_learn_dense', exist_ok=True)
    m = 0
    #
    if net_choice == "sparse":
        # Assuming `noise_images` is your list of noise images
        psnr_results = []
        output_results = []
        power_spectra = []
        for instance in range(1):
            # load masked model and net_input_list from the outdir folder
            with open(f'{outdir}/masked_model_{ino}.pkl', 'rb') as f:
                masked_model = cPickle.load(f)
            with open(f'{outdir}/net_input_list_{ino}.pkl', 'rb') as f:
                net_input_list = cPickle.load(f)
            # load the saved mask
            with open(f'{outdir}/mask_{ino}.pkl', 'rb') as f:
                mask = cPickle.load(f)
            # Example usage
            chessboard_noise = generate_chessboard_noise(
                (3, 512, 512), square_size=args.square_size, noise_level=args.noise_level)
            noise = normalize_image(chessboard_noise)
            psnr, out = train_sparse(masked_model, net_input_list, mask, noise, noise,
                                     max_step=args.max_steps, show_every=args.show_every, device=args.device_id)
            print(out.shape)
            power_spectrum = compute_fourier_power_spectrum(out)
            power_spectra.append(power_spectrum)
        avg_spectrum = average_power_spectrum(power_spectra)

        # or choose a colormap suitable for your data
        plt.imshow(out.transpose(1, 2, 0))
        plt.axis('off')  # To not display axes
        plt.imsave('average_power_spectrum_sparse.png', out.transpose(1, 2, 0))
        torch.cuda.empty_cache()
        print("Experiment done for sparse network")
    elif net_choice == "dense":
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
            act_fun='LeakyReLU').type(dtype)
        power_spectra = []
        for instance in range(1):
            chessboard_noise = generate_chessboard_noise(
                (3, 512, 512), square_size=args.square_size, noise_level=args.noise_level)
            noise = normalize_image(chessboard_noise)
            # or choose a colormap suitable for your data
            plt.imshow(noise.transpose(1, 2, 0))
            plt.axis('off')
            plt.imsave('noise_chessboard.png', noise.transpose(1, 2, 0))
            psnr, out = train_dense(net, net_input, noise, noise, max_step=args.max_steps,
                                    show_every=args.show_every, device=args.device_id)
            print(out.shape)
            power_spectrum = compute_fourier_power_spectrum(out)
            print(power_spectrum.shape)
            power_spectra.append(power_spectrum)
        avg = average_power_spectrum(power_spectra)
        avg_spectrum = np.array(avg, dtype=float)
        print(avg_spectrum)

        plt.imshow(out.transpose(1, 2, 0))
        plt.axis('off')  # To not display axes
        plt.imsave('average_power_spectrum_dense.png', out.transpose(1, 2, 0))

    elif net_choice == "deep_decoder":
        k = 5
        power_spectra = []
        for instance in range(1):
            # base_noise = np.random.normal(scale=sigma, size=img_np.shape) # initial random Gaussian noise
            # variance = args.var # different variances for Gaussian smoothing
            # noise = normalize_image(generate_noise_images(base_noise, variance))
            chessboard_noise = generate_chessboard_noise(
                (3, 512, 512), square_size=args.square_size, noise_level=args.noise_level)
            noise = normalize_image(chessboard_noise)
            psnr, out = train_deep_decoder(
                k, noise, noise, max_step=args.max_steps, show_every=args.show_every, device=args.device_id)
            print(out.shape)
            power_spectrum = compute_fourier_power_spectrum(out)
            print(power_spectrum.shape)
            # plt.imshow(power_spectrum)  # or choose a colormap suitable for your data
            # plt.axis('off')  # To not display axes
            # plt.imsave('average_power_spectrum.png',power_spectrum)
            power_spectra.append(power_spectrum)
        avg = average_power_spectrum(power_spectra)
        avg_spectrum = np.array(avg, dtype=float)
        print(avg_spectrum)

        # # Save the 'out' image
        plt.imshow(out.transpose(1, 2, 0))  # Adjust colormap as needed
        plt.axis('off')
        plt.imsave('average_power_spectrum_deepdec.png',
                   out.transpose(1, 2, 0))


    # plot_psnr(psnr_lists)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image denoising using DIP")

    parser.add_argument("--images", type=str,
                        default=["Lena512rgb"], help="which image to denoise")
    parser.add_argument("--lr", type=float,  default=1e-2,
                        help="the learning rate")
    parser.add_argument("--max_steps", type=int, default=40000,
                        help="the maximum number of gradient steps to train for")
    parser.add_argument("--reg", type=float, default=0.05,
                        help="if regularization strength of igr")
    parser.add_argument("--sigma", type=float, default=0.1, help="noise-level")
    parser.add_argument("--num_layers", type=int,
                        default=6, help="number of layers")
    parser.add_argument("--show_every", type=int,
                        default=1000, help="show_every")
    parser.add_argument("--device_id", type=int, default=0,
                        help="specify which gpu")
    parser.add_argument("--beta", type=float, default=0,
                        help="momentum for sgd ")
    parser.add_argument("--decay", type=float, default=0, help="weight decay")
    parser.add_argument("--ino", type=int, default=4, help="image index ")
    parser.add_argument("--mask_opt", type=str,
                        default="det", help="mask type")
    parser.add_argument("--noise_steps", type=int,
                        default=80000, help="numvere of steps for noise")
    parser.add_argument("--kl", type=float, default=1e-9,
                        help="regularization strength of kl")
    parser.add_argument("--prior_sigma", type=float,
                        default=-1.3, help="prior mean")
    parser.add_argument("--var", type=float, default=0.0,
                        help="variance of gaussian kernel")
    parser.add_argument("--net_choice", type=str,
                        default="sparse", help="sparse of dense")
    parser.add_argument("--square_size", type=int, default=8,
                        help="Size of the squares in the checkerboard pattern")
    parser.add_argument("--noise_level", type=float, default=0.3,
                        help="Noise level for the chessboard pattern")

    args = parser.parse_args()

    main(images=args.images, lr=args.lr, max_steps=args.max_steps,
         reg=args.reg, sigma=args.sigma, num_layers=args.num_layers,
         show_every=args.show_every, beta=args.beta, device_id=args.device_id, ino=args.ino,
         weight_decay=args.decay, mask_opt=args.mask_opt, noise_steps=args.noise_steps,
         kl=args.kl, prior_sigma=args.prior_sigma, var=args.var, net_choice=args.net_choice)
