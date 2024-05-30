

from __future__ import print_function
import torch
import math
import numpy as np
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
from utils.quant import *
from models import *
#from DIP_quant.utils.quant import *
from models.cnn import cnn
import torch.optim
import time
#from skimage.measure import compare_psnr
from utils.inpainting_utils import * 
import pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from sam import SAM

# def make_mask(model):
#     mask = []
#     for param in model.parameters(): 
#         mask.append(torch.ones_like(param.data, requires_grad=False))
#     return mask

def count_nonzero(model, mask):
    nonzero_model = 0
    nonzero_mask = 0
    total = 0
    mask_count = 0

    for i, param in enumerate(model.parameters()):
        # Calculate non-zeros in the model parameters and the mask
        nonzero_model += torch.count_nonzero(param.data)
        nonzero_mask += torch.count_nonzero(mask[i])

        # Calculate total elements in the model parameters and the mask
        total += param.data.numel()
        mask_count += mask[i].numel()
    ##print a statement that prints the percentage of nonzero weights in the model and the mask
    print(f"Non-zero model percentage: {nonzero_model / total * 100}%, Non-zero mask percentage: {nonzero_mask / mask_count * 100}%") 


# Function to prune the model
def prune_magnitude_global(model, mask, percent):
    # Calculate percentile value
    importance_score = []
    for name, param in model.named_parameters():
        importance_score.extend(param.data.abs().view(-1))

    importance_score = torch.stack(importance_score)
    weight_indices = torch.arange(len(mask), device=mask.device)[mask > 0]
    permuted_indices = torch.argsort(importance_score[mask > 0])
    percentile_value = np.quantile(importance_score[mask > 0].cpu().numpy(), percent)
    print(f'Pruning with threshold : {percentile_value}')

    num_to_prune = math.ceil(mask.sum() * percent)
    indices_to_prune = permuted_indices[:num_to_prune]
    mask[weight_indices[indices_to_prune]] = 0.0

    # Update model weights
    start_idx = 0
    for param in model.parameters():
        end_idx = start_idx + param.numel()
        param.data *= mask[start_idx:end_idx].view(param.shape)
        start_idx = end_idx

    return mask


def prune_random_global(model, mask, percent):
    active_weights_indices = torch.where(mask > 0)[0]
    num_active_weights = active_weights_indices.numel()
    num_to_prune = math.ceil(num_active_weights * percent)

    indices_to_prune = active_weights_indices[torch.randperm(num_active_weights)[:num_to_prune]]
    mask[indices_to_prune] = 0.0

    start_idx = 0
    for param in model.parameters():
        end_idx = start_idx + param.numel()
        param.data *= mask[start_idx:end_idx].view(param.shape)
        start_idx = end_idx

    return mask


def prune_random_local(model, mask, percent):
    start_idx = 0
    for param in model.parameters():
        end_idx = start_idx + param.numel()
        mask_flat = mask[start_idx:end_idx]
        active_weights_indices = torch.where(mask_flat > 0)[0]
        num_active_weights = active_weights_indices.numel()
        num_to_prune = math.ceil(num_active_weights * percent)

        indices_to_prune = active_weights_indices[torch.randperm(num_active_weights)[:num_to_prune]]
        mask_flat[indices_to_prune] = 0.0
        param.data *= mask_flat.view(param.shape)
        start_idx = end_idx

    return mask


def snip_prune(model, mask, net_input, img_var, noise_var, percent):
    # Compute gradients
    model.zero_grad()
    mse = torch.nn.MSELoss()
    img_var = np_to_torch(img_var).type(dtype)
    noise_var = np_to_torch(noise_var).type(dtype)
    out = model(net_input)
    total_loss = mse(out, noise_var)
    total_loss.backward()

    grads = []
    for param in model.parameters():
        if param.requires_grad:
            grads.append(param.grad.data.view(-1).abs())

    all_grads = torch.cat(grads)
    num_to_prune = int(len(all_grads) * percent)
    threshold, _ = torch.topk(all_grads, num_to_prune, sorted=True)
    acceptable_grad = threshold[-1]

    # Update mask
    start_idx = 0
    for param in model.parameters():
        if param.requires_grad:
            end_idx = start_idx + param.numel()
            param_grad = all_grads[start_idx:end_idx]
            mask_segment = (param_grad > acceptable_grad).int()
            mask[start_idx:end_idx] = mask_segment
            param.data *= mask_segment.view(param.shape)
            start_idx = end_idx

    return mask

def grasp_prune(net, mask, net_input, img_var, noise_var, percent):
    eps = 1e-10
    keep_ratio = 1 - percent
    old_net = copy.deepcopy(net)
    net.zero_grad()
    weights = [param for param in net.parameters() if param.requires_grad]
    device = next(net.parameters()).device
    # Compute the loss and gradients
    mse = torch.nn.MSELoss()
    img_var = np_to_torch(img_var).to(device)
    noise_var = np_to_torch(noise_var).to(device)
    out = net(net_input)
    loss = mse(out, noise_var)
    grad_w = torch.autograd.grad(loss, weights, create_graph=True)

    # Compute the Hessian-gradient product (second order derivatives)
    z = 0
    for g, w in zip(grad_w, weights):
        z += (g * w).sum()
    z.backward()

    # Collect the gradients after the second backward pass
    grads = {w: w.grad.data for w in weights}

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1 - keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    acceptable_score = threshold[-1]
    keep_masks = {w: ((g / norm_factor) <= acceptable_score).float() for w, g in grads.items()}

    # Apply the computed mask to the parameters
    start_idx = 0
    for param in net.parameters():
        if param.requires_grad:
            end_idx = start_idx + param.numel()
            param_mask = keep_masks[param].view(-1)
            mask[start_idx:end_idx] = param_mask
            param.data *= param_mask.view(param.shape)
            start_idx = end_idx

    return mask



def synflow_prune(model, mask, net_input, percent):
    # Forward and backward pass with random inputs
    model.zero_grad()
    output = model(net_input)
    fake_loss = output.sum()
    fake_loss.backward()

    scores = []
    for param in model.parameters():
        if param.requires_grad:
            scores.append(param.grad.data.view(-1).abs())

    all_scores = torch.cat(scores)
    num_to_prune = int(len(all_scores) * percent)
    threshold, _ = torch.topk(all_scores, num_to_prune, sorted=True)
    acceptable_score = threshold[-1]

    # Update mask and apply it to model's parameters
    start_idx = 0
    for param in model.parameters():
        if param.requires_grad:
            end_idx = start_idx + param.numel()
            param_score = all_scores[start_idx:end_idx]
            mask_segment = (param_score > acceptable_score).int()
            mask[start_idx:end_idx] = mask_segment
            param.data *= mask_segment.view(param.shape)
            start_idx = end_idx

    return mask



# Function to get a mask of the pruned model
def get_pruning_mask(model):
    mask = []
    for param in model.parameters():
        if param.requires_grad:
            param_mask = (param.data != 0).int()
            mask.append(param_mask.view(-1))
    return torch.cat(mask)


def train_and_prune_model(model, net_input, img_np, img_noisy_np, prune_type="random", max_step=40000,percent=0.3, prune_epoch=5000,learning_rate=1e-2, mask=None, device='cuda:0'):
    img_var = np_to_torch(img_np)
    noise_var = np_to_torch(img_noisy_np)
    model = model.to(device)
    img_var = img_var.to(device)
    noise_var = noise_var.to(device)
    net_input = net_input.to(device)
    mask=  get_pruning_mask(model) 

    mse = torch.nn.MSELoss().type(dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    psnr_lists = []

    for epoch in range(max_step):
        optimizer.zero_grad()
        out = model(net_input)
        total_loss = mse(out, noise_var)
        total_loss.backward()

        if epoch % prune_epoch == 0 and epoch!=0:
            if prune_type == 'magnitude':
                mask = prune_magnitude_global(model, mask, percent)
            else:
                mask = prune_random_global(model, mask, percent)  
            print_nonzeros(model)      

            # Apply mask to gradients
        k = 0
        for param in model.parameters():
            if param.requires_grad:
                t = len(param.view(-1))
                param.grad.data = param.grad.data * mask[k:(k+t)].view(param.grad.data.shape)
                k += t

        optimizer.step()

        # Calculate and store PSNR every 10000 epochs
        if epoch % 200 == 0:
            out_np = out.detach().cpu().numpy()[0]
            img_np = img_var.detach().cpu().numpy()
            psnr_gt = compare_psnr(img_np, out_np)
            psnr_lists.append(psnr_gt)
            print(f"Epoch: {epoch}, Loss: {total_loss.item()}, PSNR: {psnr_gt}")

    return psnr_lists, out.detach().cpu().numpy()  



# Function to train the model (with pruning if a mask is provided)
def train_model(model, net_input, img_var, noise_var, max_step=40000, learning_rate=0.001, mask=None, device='cuda:0'):
    model = model.to(device)
    img_var = img_var.to(device)
    noise_var = noise_var.to(device)
    net_input = net_input.to(device)

    mse = torch.nn.MSELoss().type(dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    psnr_lists = []

    for epoch in range(max_step):
        optimizer.zero_grad()
        out = model(net_input)
        total_loss = mse(out, noise_var)
        total_loss.backward()

        if mask is not None:
            # Apply mask to gradients
            k = 0
            for param in model.parameters():
                if param.requires_grad:
                    t = len(param.view(-1))
                    param.grad.data = param.grad.data * mask[k:(k+t)].view(param.grad.data.shape)
                    k += t

        optimizer.step()

        # Calculate and store PSNR every 10000 epochs
        if epoch % 10000 == 0:
            out_np = out.detach().cpu().numpy()[0]
            img_np = img_var.detach().cpu().numpy()
            psnr_gt = compare_psnr(img_np, out_np)
            psnr_lists.append(psnr_gt)
            print(f"Epoch: {epoch}, Loss: {total_loss.item()}, PSNR: {psnr_gt}")

    return psnr_lists, out.detach().cpu().numpy()

# Main iterative pruning loop
def iterative_pruning(model, net_input, img_var, noise_var, pruning_percentage, iter_prune,num_epoch, device='cuda:0'):
    psnr_history = []
    model = model.to(device)  # Move model to the specified device
    #net_input, img_var, noise_var = net_input.to(device), img_var.to(device), noise_var.to(device)  # Move inputs to the device

    mask = get_pruning_mask(model).to(device)  # Ensure mask is on the same device
    model_init = copy.deepcopy(model).to(device)  # Deepcopy model to the device

    for iteration in range(iter_prune):
        print(f"Pruning Iteration: {iteration+1}/{iter_prune}")
        #print_nonzeros(model)

        if iteration > 0:
            prune_magnitude_global(model, mask, pruning_percentage)  # Pass device to the function
            mask = get_pruning_mask(model).to(device)
            original_initialization(model, mask, model_init)  # Pass device to the function

        psnr_list, _ = train_sparse(model, net_input, mask, img_var, noise_var, max_step=num_epoch,device=device)  # Pass device to the function
        psnr_history.append(psnr_list)

    return model, mask, psnr_history

def original_initialization(model, mask, model_init):
    start_idx = 0
    for param, param_init in zip(model.parameters(), model_init.parameters()):
        end_idx = start_idx + param.numel()
        mask_segment = mask[start_idx:end_idx].view(param.shape)
        param.data = mask_segment * param_init.data
        start_idx = end_idx
    return

def snip_prune_local(model, mask, net_input, img_var, noise_var, percent):
    # Compute gradients
    model.zero_grad()
    mse = torch.nn.MSELoss()
    img_var = np_to_torch(img_var).type(dtype)
    noise_var = np_to_torch(noise_var).type(dtype)
    out = model(net_input)
    total_loss = mse(out, noise_var)
    total_loss.backward()

    start_idx = 0
    for param in model.parameters():
        if param.requires_grad:
            # Flatten and get the absolute values of the gradients
            param_grad = param.grad.data.view(-1).abs()

            # Calculate the number of elements to prune in this layer
            num_to_prune_local = int(len(param_grad) * percent)

            # Find the local threshold value for pruning
            threshold_local, _ = torch.topk(param_grad, num_to_prune_local, sorted=True)
            acceptable_grad_local = threshold_local[-1]

            # Update the mask and the parameter data
            mask_segment = (param_grad > acceptable_grad_local).int()
            param_mask_flat = mask_segment.view(-1)
            param.data *= param_mask_flat.view(param.shape)

            # Update the global mask
            end_idx = start_idx + param.numel()
            mask[start_idx:end_idx] = param_mask_flat
            start_idx = end_idx

    return mask



def grasp_prune_local(net, mask, net_input, img_var, noise_var, percent):
    eps = 1e-10
    keep_ratio = 1 - percent
    old_net = copy.deepcopy(net)
    net.zero_grad()
    weights = [param for param in net.parameters() if param.requires_grad]
    device = next(net.parameters()).device

    # Compute the loss and gradients
    mse = torch.nn.MSELoss()
    img_var = np_to_torch(img_var).to(device)
    noise_var = np_to_torch(noise_var).to(device)
    out = net(net_input)
    loss = mse(out, noise_var)
    grad_w = torch.autograd.grad(loss, weights, create_graph=True)

    # Compute the Hessian-gradient product
    z = 0
    for g, w in zip(grad_w, weights):
        z += (g * w).sum()
    z.backward()

    start_idx = 0
    for w, g in zip(weights, grad_w):
        if w.requires_grad:
            # Calculate the score for each parameter
            grad = g.data.view(-1).abs()
            hessian = w.grad.data.view(-1).abs()
            score = grad / (hessian + eps)

            # Normalize the score
            norm_factor = torch.abs(torch.sum(score)) + eps
            score.div_(norm_factor)

            # Determine local pruning threshold
            num_params_to_rm = int(len(score) * percent)
            threshold, _ = torch.topk(score, num_params_to_rm, sorted=True)
            acceptable_score = threshold[-1]

            # Update mask
            param_mask = (score > acceptable_score).float()
            w.data *= param_mask.view(w.shape)

            # Update global mask
            end_idx = start_idx + w.numel()
            mask[start_idx:end_idx] = param_mask.view(-1)
            start_idx = end_idx

    return mask




def synflow_prune_local(model, mask, net_input, percent):
    # Forward and backward pass with random inputs
    model.zero_grad()
    output = model(net_input)
    fake_loss = output.sum()
    fake_loss.backward()

    start_idx = 0
    for param in model.parameters():
        if param.requires_grad:
            # Calculate scores
            score = param.grad.data.view(-1).abs()

            # Determine local pruning threshold
            num_to_prune = int(len(score) * percent)
            threshold, _ = torch.topk(score, num_to_prune, sorted=True)
            acceptable_score = threshold[-1]

            # Update mask and apply locally
            mask_segment = (score > acceptable_score).int().view(param.shape)
            param.data *= mask_segment

            # Update global mask
            end_idx = start_idx + param.numel()
            mask[start_idx:end_idx] = mask_segment.view(-1)
            start_idx = end_idx

    return mask

