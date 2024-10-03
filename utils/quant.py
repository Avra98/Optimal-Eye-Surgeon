from __future__ import print_function
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from models import *
from models.decoder import decodernw
from utils import *
from utils.denoising_utils import *
from utils.common_utils import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
logger = logging.getLogger('main')

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


def gumbel_softmax_multi(logits, temperature=0.2):
    # Add a column for 1-p1-p2, clamped to prevent log(0)
    logits = torch.cat(
        (logits, 1 - logits.sum(dim=1, keepdim=True).clamp(min=1e-20, max=1-1e-20)), dim=1)

    # Gumbel noise; explicitly without gradients, clamped to prevent log(0)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(
        logits).clamp(min=1e-20, max=1-1e-20))).requires_grad_(False)

    # Logits; clamped to prevent log(0)
    y = (torch.log(logits.clamp(min=1e-20, max=1-1e-20)) +
         gumbel_noise) / temperature

    # Softmax output
    softmax_output = F.softmax(y, dim=-1)

    return softmax_output


def soft_quantize(logits, q, temperature=0.2):
    soft_samples = gumbel_softmax_multi(logits, temperature)
    with torch.no_grad():
        quant_values = [torch.tensor([1 - i / (q - 1)])
                        for i in range(q - 1)] + [torch.tensor([0.0])]
        quant_values = torch.cat(quant_values).to(logits.device)
    quantized_output = torch.sum(soft_samples * quant_values, dim=-1)
    return quantized_output

def quant_initialization(model, q=3):
    device = next(model.parameters()).device

    w0, num_params = [], 0
    for layer, param in enumerate(model.parameters()):
        w0.append(param.data.view(-1).detach().clone())
    w0 = torch.cat(w0)
    p = nn.Parameter(inverse_sigmoid(1/q)*torch.ones([q-1, w0.size(0)]).to(device), requires_grad=True)
    return w0, p

def attach_quantization_probabilities_to_model(model, p):
    assert len(p.shape) == 1
    assert len(p) == sum(p.numel() for p in model.parameters())

    device = next(model.parameters()).device
    p.to(device)

    module_dict = dict(model.named_modules())

    start = 0
    for name, param in model.named_parameters():
        end = start + param.numel()
        param_p = p[start:end].view_as(param)
        
        module_name, param_name = name.rsplit('.', 1)
        module = module_dict[module_name]
        setattr(module, f'{param_name}_p', param_p)

        start = end

def learn_quantization_probabilities_dip(model, net_input, img_var, noise_var, num_steps, lr, q=3, kl=1e-5, prior_sigma=torch.tensor(0.0), sparsity=0.5, show_every=1000):
    """
    Args:
        model (nn.Module): The PyTorch model to be quantized.
        net_input (torch.Tensor): The input tensor for the model.
        img_var (np.ndarray): The ground truth image as a NumPy array.
        noise_var (np.ndarray): The noisy image as a NumPy array.
        num_steps (int): The number of training steps.
        lr (float): The learning rate for the optimizer.
        ino (int): An identifier for the experiment.
        q (int, optional): The number of quantization levels. Defaults to 3.
        kl (float, optional): The weight for the KL divergence regularization term. Defaults to 1e-5.
        prior_sigma (torch.tensor, optional): The prior sigma value for the KL divergence. Defaults to torch.tensor(0.0).
        sparsity (float, optional): The target sparsity level (percentage of elements to keep active). Defaults to 0.5.
        show_every (int, optional): The frequency at which to print training statistics. Defaults to 1000.

    Returns:
        tuple: A tuple containing:
            torch.Tensor: The learned quantization probabilities (p).
            list: A list of quantization loss values recorded during training.
    """
    device = next(model.parameters()).device
    mse = torch.nn.MSELoss()
    img_var = np_to_torch(img_var).to(device)
    noise_var = np_to_torch(noise_var).to(device)

    # Initialize quantization probabilities (p) and make sure they require gradients
    w0, p = quant_initialization(model, q)
    p.requires_grad_(True)
    optimizer_p = torch.optim.Adam([p], lr=lr)
    # TODO: add learning rate scheduler ?
    prior = sigmoid(prior_sigma)

    all_logits = []
    quant_loss = []

    for iteration in range(num_steps):

        # make a copy of the model and freeze the weights
        model_copy = copy.deepcopy(model)
        for param in model_copy.parameters():
            param.requires_grad = False

        # Update quantization probabilities using gradient descent
        optimizer_p.zero_grad()
        k = 0

        # TOOD: I feel like p can be reshaped and used directly with torch pruning
        for i, param in enumerate(model_copy.parameters()):
            t = len(param.view(-1))
            logits = p[:, k:(k+t)].t()
            quantized_weights = soft_quantize(torch.sigmoid(logits), q, temperature=0.2)
            param.mul_(quantized_weights.view(param.data.shape))
            k += t

        # Forward pass after quantization
        output = model_copy(net_input)
        # Compute the regularization term based on the KL divergence
        reg = (torch.sigmoid(p) * torch.log((torch.sigmoid(p)+1e-6)/prior) +
               (1-torch.sigmoid(p)) * torch.log((1-torch.sigmoid(p)+1e-6)/(1-prior))).sum()

        # Compute loss based on the dissimilarity between quantized model output and noisy image
        quantization_loss = mse(output, noise_var) + kl*reg
        quantization_loss.backward()
        optimizer_p.step()

        # Update quantization probabilities using gradient descent
        with torch.no_grad():
            if iteration % show_every == 0:
                logger.info("iteration: %s   quantization_loss: %s   p mean: %s",
                            iteration, quantization_loss.item(), p.mean().item())
                quant_loss.append(quantization_loss.item())

        if iteration == num_steps - 1:
            logits_flat = torch.sigmoid(p).view(-1).cpu().detach().numpy()
            all_logits.extend(logits_flat)

    return p, quant_loss, all_logits

def draw_one_mask(logits, model, net_input):
    k = 0
    for i, param in enumerate(model.parameters()):
        t = len(param.view(-1))
        log = logits[:, k:(k+t)].t()
        quantized_weights = soft_quantize(
            torch.sigmoid(log), q=2, temperature=0.2)
        param.data = param.data * quantized_weights.view(param.data.shape)
        k += t
    output = model(net_input)
    return output


def draw_multiple_masks(logits, model, net_input, num_masks=10):
    hard_quantized_images = []

    for _ in range(num_masks):
        k = 0
        for i, param in enumerate(model.parameters()):
            t = len(param.view(-1))
            log = logits[:, k:(k+t)].t()
            quantized_weights = soft_quantize(
                torch.sigmoid(log), q=2, temperature=0.2)
            param.data = param.data * quantized_weights.view(param.data.shape)
            k += t
        output = model(net_input)
        hard_quantized_images.append(output)

    average_output = torch.mean(torch.stack(hard_quantized_images), dim=0)
    return average_output


def deterministic_rounding(model, net_input):
    output = model(net_input)
    return output



def train_sparse(masked_model, net_input, mask, img_var, noise_var, learning_rate=0.01, max_step=40000, show_every=200, lr_step=100000, lr_gamma=0.1, device=None):
    """Trains a sparse model using masked backpropagation.

    This function trains a PyTorch model with sparsity enforced by a provided mask.
    During backpropagation, gradients are only updated for parameters corresponding to
    non-zero elements in the mask.

    Args:
        masked_model (nn.Module): The sparse model to be trained.
        net_input (torch.Tensor): The input tensor for the model.
        mask (torch.Tensor): The mask tensor defining sparsity (1 for active elements, 0 for inactive).
        img_var (np.ndarray): The ground truth image as a NumPy array.
        noise_var (np.ndarray): The noisy image as a NumPy array.
        learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.01.
        max_step (int, optional): The maximum number of training steps. Defaults to 40000.
        show_every (int, optional): The frequency at which to print training statistics (epoch, loss, PSNR). Defaults to 200.
        lr_step (int, optional): The step at which to decay the learning rate. Defaults to 100000.
        lr_gamma (float, optional): The learning rate decay factor. Defaults to 0.1.
        device (str, optional): The device to use for training (CPU or GPU). Defaults to None (determined automatically).

    Returns:
        tuple: A tuple containing:
            list: A list of PSNR values recorded during training.
            numpy.ndarray: The final output image (trained model's reconstruction).
    """
    # Setting the device for the model and tensors
    masked_model = masked_model.to(device)
    img_var = np_to_torch(img_var).to(device)
    noise_var = np_to_torch(noise_var).to(device)
    net_input = net_input.to(device)
    mask = mask.to(device)

    mse = torch.nn.MSELoss()
    psnr_lists = []
    ssim_lists = []
    print_nonzeros(masked_model)
    # Define the optimizer
    optimizer = torch.optim.Adam(masked_model.parameters(), lr=learning_rate)

    # Forward pass to compute initial PSNR
    for epoch in range(max_step):
        optimizer.zero_grad()
        out = masked_model(net_input)

        total_loss = mse(out, noise_var)
        with torch.no_grad():
            if epoch == 0:
                logger.info("total_loss is: %s", total_loss)

        total_loss.backward()

        # Zero gradients for the masked parameters
        # TODO: I think this can be done more efficiently with perhaps
        # disabling gradients for the masked parameters 
        k = 0
        for param in masked_model.parameters():
            t = len(param.view(-1))
            param.grad.data = param.grad.data * \
                mask[k:(k+t)].view(param.grad.data.shape)
            k += t

        optimizer.step()

        if epoch % show_every == 0:
            # Calculating PSNR
            out_np = out.detach().cpu().numpy()
            img_np = img_var.detach().cpu().numpy()
            noisy_np = noise_var.detach().cpu().numpy()
            ssim_gt = structural_similarity(img_np[0], out_np[0],
                                            channel_axis=0, data_range=img_np.max()-img_np.min())
            ssim_noisy = structural_similarity(noisy_np[0], out_np[0], 
                                               channel_axis=0, data_range=noisy_np.max()-noisy_np.min())
            ssim_lists.append(ssim_gt)
            psnr_gt = compare_psnr(img_np, out_np)
            psnr_noisy = compare_psnr(noisy_np, out_np)
            psnr_lists.append(psnr_gt)

            # Print epoch, loss and PSNR
            logger.info("epoch: %s  loss: %s  SSIM: %s  SSIM_noisy %s  PSNR: %s  PSNR_noisy %s", 
                        epoch, total_loss.item(), ssim_gt, ssim_noisy, psnr_gt, psnr_noisy)

    return ssim_lists, psnr_lists, out_np


def train_dense(net, net_input,  img_var, noise_var, learning_rate=1e-3, max_step=40000, show_every=1000, lr_step=100000, lr_gamma=0.1, device='cuda:0'):
    # Setting the device for the model and tensors
    net = net.to(device)
    img_var = np_to_torch(img_var).to(device)
    noise_var = np_to_torch(noise_var).to(device)
    net_input = net_input.to(device)

    mse = torch.nn.MSELoss()
    psnr_lists = []
    print_nonzeros(net)
    # Define the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Forward pass to compute initial PSNR
    with torch.no_grad():
        initial_out = net(net_input)
        initial_psnr = compare_psnr(
            img_var.detach().cpu().numpy(), initial_out.detach().cpu().numpy())
        print("Initial PSNR of output image is: ", initial_psnr)

    for epoch in range(max_step):
        optimizer.zero_grad()
        out = net(net_input)
        total_loss = mse(out, noise_var)
        total_loss.backward()

        optimizer.step()

        if epoch % show_every == 0:
            # Calculating PSNR
            out_np = out.detach().cpu().numpy()
            img_np = img_var.detach().cpu().numpy()
            psnr_gt = compare_psnr(img_np, out_np)
            psnr_lists.append(psnr_gt)
            # print_nonzeros(masked_model)

            # Print epoch, loss and PSNR
            print("epoch: ", epoch, "loss: ",
                  total_loss.item(), "PSNR: ", psnr_gt)

    return psnr_lists, out_np


def train_deep_decoder(k, img_var, noise_var, learning_rate=0.01, max_step=40000, show_every=1000, device='cuda:1'):
    output_depth = img_var.shape[0]
    img_var = np_to_torch(img_var).to(device)
    noise_var = np_to_torch(noise_var).to(device)
    mse = torch.nn.MSELoss()
    num_channels = [128]*k
    totalupsample = 2**len(num_channels)
    width = int(img_var.shape[2]/totalupsample)
    height = int(img_var.shape[2]/totalupsample)
    shape = [1, num_channels[0], width, height]
    net_input = Variable(torch.zeros(shape))
    net_input.data.uniform_()
    net_input.data *= 1./10
    # net_input = net_input.type(dtype)
    net = decodernw(output_depth, num_channels_up=num_channels,
                    upsample_first=True)
    # print total number of paramter in net
    s = sum([np.prod(list(p.size())) for p in net.parameters()])
    print('Number of params in decoder is: %d' % s)
    # Define the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    psnr_lists = []
    for epoch in range(max_step):
        optimizer.zero_grad()
        out = net(net_input)
        total_loss = mse(out, noise_var)
        total_loss.backward()
        optimizer.step()
        if epoch % show_every == 0:
            # Calculating PSNR
            out_np = out.detach().cpu().numpy()
            img_np = img_var.detach().cpu().numpy()
            psnr_gt = compare_psnr(img_np, out_np)
            psnr_lists.append(psnr_gt)
            print("epoch: ", epoch, "loss: ",
                  total_loss.item(), "PSNR: ", psnr_gt)
    return psnr_lists, out_np


def print_nonzeros(model):
    """Prints the number of nonzero elements and pruning rate of a model's parameters.

    Args:
        model: PyTorch model to analyze.

    Returns:
        float: Percentage of nonzero parameters (rounded to 1 decimal place).
    """
    nonzero = total = 0
    for name, param in model.named_parameters():
        nz_count = int(torch.count_nonzero(param.data))
        total_params = param.data.numel()
        nonzero += nz_count
        total += total_params
        logger.debug(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} \
            ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count:7} | shape = {param.data.shape}')

    logger.info(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, \
                Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return (round((nonzero/total)*100, 1))


def inverse_sigmoid(x):
    return torch.log(torch.tensor(x) / torch.tensor(1 - x))


def load_image(train_folder, image_name, sigma):
    train_noisy_folder = f'{train_folder}/train_noisy_{sigma}'
    os.makedirs(train_noisy_folder, exist_ok=True)

    file_path = os.path.join(train_folder, f'{image_name}.png')
    filename = os.path.splitext(os.path.basename(file_path))[0]
    img_pil = Image.open(file_path)
    img_pil = resize_and_crop(img_pil, max(img_pil.size))
    img_np = pil_to_np(img_pil)
    img_noisy_np = np.clip(
        img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)

    img_noisy_pil = np_to_pil(img_noisy_np)
    img_noisy_pil.save(os.path.join(train_noisy_folder, filename + '.png'))

    noisy_psnr = compare_psnr(img_np, img_noisy_np)
    return img_np, img_noisy_np, noisy_psnr
