from __future__ import print_function
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import copy
import torch
import torch.nn as nn
import numpy as np
import random
import os
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from models import *
from models.decoder import decodernw
from utils import *
from utils.denoising_utils import *
from utils.common_utils import *


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# TODO modernize device here too


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


def learn_quantization_probabilities_dip(model, net_input, img_var, noise_var, num_steps, lr, ino, q=3, kl=1e-5, prior_sigma=torch.tensor(0.0), sparsity=0.5, show_every=1000):
    """Learns quantization probabilities using a deep inverse prior (DIP) approach.

    This function trains a model to learn quantization probabilities (p) for a specific sparsity level.
    The training process uses a deep inverse prior (DIP) strategy to optimize the p values while
    considering a mean squared error (MSE) loss between the model's output and a noisy image,
    along with a regularization term based on the Kullback-Leibler (KL) divergence between p and a prior distribution.

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
    _, p = quant_initialization(model, q)
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
        for i, param in enumerate(model_copy.parameters()):
            t = len(param.view(-1))
            logits = p[:, k:(k+t)].t()
            quantized_weights = soft_quantize (
                torch.sigmoid(logits), q, temperature=0.2)
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
                print("iteration: ", iteration, "quantization_loss: ",
                      quantization_loss.item())
                quant_loss.append(quantization_loss.item())
                print("p mean is:", p.mean())

        if iteration == num_steps - 1:
            logits_flat = torch.sigmoid(p).view(-1).cpu().detach().numpy()
            all_logits.extend(logits_flat)

    os.makedirs(f'histogram_centeredl1_{ino}', exist_ok=True)

    # Plot a histogram for all the quantized weights
    plt.hist(all_logits, bins=50, alpha=0.5, label='All Layers')
    plt.title(f'Distribution of p for sparsity level {sparsity}')
    plt.xlabel('Value of p')
    plt.ylabel('Frequency')
    plt.savefig(f'histogram_centeredl1_{ino}/all_layers_histogram_q_{ino}_{sparsity}_{kl}.png')
    plt.clf()

    return p, quant_loss


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
    print_nonzeros(model)
    output = model(net_input)
    return output


def make_mask_with_sparsity(logits, sparsity=0.05):
    """Creates a mask for enforcing sparsity based on a thresholding strategy.

    This function generates a binary mask from a provided tensor (logits) to enforce a desired sparsity level.
    Elements in the logits tensor with values exceeding a threshold are set to 1 (active),
    and the remaining elements are set to 0 (inactive).

    Args:
        logits (torch.Tensor): The input tensor used to generate the mask.
        sparsity (float, optional): The target sparsity level (percentage of elements to keep active). Defaults to 0.05 (5%).

    Returns:
        torch.Tensor: The generated sparse mask (same size as the input logits).
    """
    num_elements = logits.numel()
    num_to_keep = int(sparsity * num_elements)
    print(f"Number of elements to keep: {num_to_keep}")

    # Get the threshold and top elements
    values, indices = torch.topk(logits.view(-1), num_to_keep, largest=True)
    threshold = values.min()
    print(f"Threshold value: {threshold}")

    # Identify elements equal to the threshold
    equal_to_threshold = logits.view(-1) == threshold
    num_equal_elements = equal_to_threshold.sum().item()
    print(f"Number of elements equal to threshold: {num_equal_elements}")

    # Calculate the number of elements to randomly select among equals
    num_to_randomly_select = int(max(
        0, num_to_keep - (values > threshold).sum().item()))
    print(f"Number of elements to randomly select: {num_to_randomly_select}")

    if num_to_randomly_select and num_equal_elements > num_to_randomly_select:
        print("Warning: Random selection among elements equal to the threshold is being performed to maintain sparsity.")
        equal_indices = torch.where(equal_to_threshold)[0].tolist()
        selected_indices = random.sample(equal_indices, num_to_randomly_select)
        equal_to_threshold[:] = 0  # Reset all equal elements to zero
        equal_to_threshold[selected_indices] = 1  # Set selected indices to one

    # Create sparse mask
    sparse_mask = (logits.view(-1) > threshold) | equal_to_threshold
    # sparse_mask_prob = mask_prob.view(-1) * sparse_mask
    # hard_quant = torch.round(sparse_mask_prob)

    actual_sparsity = (sparse_mask == 1).float().sum().item() / num_elements
    print(f"Actual sparsity achieved: {actual_sparsity}")

    return sparse_mask


def mask_network(mask, model):
    # Ensure mask and model are on the same device
    device = next(model.parameters()).device
    mask = mask.to(device)

    k = 0
    for param in model.parameters():
        t = len(param.view(-1))
        param.data = param.data * mask[k:(k + t)].view(param.data.shape)
        k += t

    return model


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
                print("total_loss is:", total_loss)

        total_loss.backward()

        # Adjust gradients according to the mask
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
            psnr_gt = compare_psnr(img_np, out_np)
            psnr_noisy = compare_psnr(noise_var.detach().cpu().numpy(), out_np)
            psnr_lists.append(psnr_gt)

            # Print epoch, loss and PSNR
            print("epoch: ", epoch, "loss: ", total_loss.item(),
                  "PSNR: ", psnr_gt, "PSNR_noisy: ", psnr_noisy)

    return psnr_lists, out_np


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
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} \
            ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count:7} | shape = {param.data.shape}')

    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, \
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
