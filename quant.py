from __future__ import print_function
import copy
import torch
import torch.nn as nn 
import numpy as np 
import random
import sys 
import os
import glob
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.denoising_utils import *
from models import *
from scipy.ndimage import gaussian_filter
from utils import *
import pickle as cPickle
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from sam import SAM
from utils.common_utils import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor
from models.decoder import decodernw,resdecoder
from torch.autograd import Variable
import time
import torch.nn.init as init
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


import torch.nn.functional as F

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

# def compare_psnr(img1, img2):
#     MSE = np.mean(np.abs(img1-img2)**2)
#     psnr=10*np.log10(np.max(np.abs(img1))**2/MSE)
#     return psnr 

def inverse_sigmoid(x):
    return torch.log(torch.tensor(x) / torch.tensor(1 - x))

# def gumbel_softmax_multi(logits, temperature=0.2):
#     # print("logits shape is before: ", logits.shape)

#     #num_classes = logits.shape[1]
    
#     # Add a column for 1-p1-p2
#     logits = torch.cat((logits, 1 - logits.sum(dim=1, keepdim=True)), dim=1) 
#     # print("logits shape is after: ", logits.shape)  
#     gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits))).requires_grad_(False)
#     y = (torch.log(logits) + gumbel_noise) / temperature
#     softmax_output = F.softmax(y, dim=-1)   
#     return softmax_output


def gumbel_softmax_multi(logits, temperature=0.2):
    # Add a column for 1-p1-p2, clamped to prevent log(0)
    logits = torch.cat((logits, 1 - logits.sum(dim=1, keepdim=True).clamp(min=1e-20, max=1-1e-20)), dim=1) 
    
    # Gumbel noise; explicitly without gradients, clamped to prevent log(0)
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-20, max=1-1e-20))).requires_grad_(False)
    
    # Logits; clamped to prevent log(0)
    y = (torch.log(logits.clamp(min=1e-20, max=1-1e-20)) + gumbel_noise) / temperature
    
    # Softmax output
    softmax_output = F.softmax(y, dim=-1)
    
    return softmax_output


def soft_quantize(logits, q, temperature=0.2):
    soft_samples = gumbel_softmax_multi(logits, temperature)
    #print("soft_samples is: ", soft_samples.shape)
    with torch.no_grad():
        quant_values = [torch.tensor([1 - i / (q - 1)]) for i in range(q - 1)] + [torch.tensor([0.0])]
        quant_values = torch.cat(quant_values).to(logits.device)
    # print("quant_values is: ", quant_values)
    quantized_output = torch.sum(soft_samples * quant_values, dim=-1)
    return quantized_output

def quant_initialization(model,prior_sigma,q=3):
    device = next(model.parameters()).device
    
    w0, num_params = [], 0
    for layer, param in enumerate(model.parameters()):
        w0.append(param.data.view(-1).detach().clone())
        # num_params += mask[layer].sum()
    # num_layer = layer + 1
    w0 = torch.cat(w0) 
    #p = nn.Parameter(torch.zeros_like(w0), requires_grad=True)
    p =  nn.Parameter(inverse_sigmoid(1/q)*torch.ones([q-1, w0.size(0)]).to(device), requires_grad=True)
    # p =  nn.Parameter(-5*torch.ones([q-1, w0.size(0)]).to(device), requires_grad=True)
    # with torch.no_grad():  
    #     p[0, :].fill_((1.5/q)-0.25)
    #     print("p elementS are",p[:,0])
    prior = sigmoid(prior_sigma)
    return w0, p,prior


def learn_quantization_probabilities_dip(model, net_input, img_var, noise_var, num_steps, lr, ino, q=3, kl=1e-5, prior_sigma=0.0, sparsity=0.5):
    device = next(model.parameters()).device
    mse = torch.nn.MSELoss().type(dtype)
    img_var = np_to_torch(img_var).type(dtype)
    noise_var = np_to_torch(noise_var).type(dtype)
    
    # Initialize quantization probabilities (p) and make sure they require gradients
    _, p, _ = quant_initialization(model, 1.0, q)
    numel = p.numel()
    p.requires_grad_(True)
    optimizer_p = torch.optim.Adam([p], lr=lr)
    num_realizations =5
    prior = sigmoid(prior_sigma)

    all_logits = []
    quant_loss =[]

    for epoch in range(num_steps):
        
        model_copy = copy.deepcopy(model)
        for param in model_copy.parameters():
            param.requires_grad = False
        quantization_loss_accum = 0.0

#        for realization in range(num_realizations):
        optimizer_p.zero_grad()
        k = 0
        for i, param in enumerate(model_copy.parameters()):
            t = len(param.view(-1))
            logits = p[:, k:(k+t)].t()
            quantized_weights = soft_quantize(torch.sigmoid(logits), q, temperature=0.2)
            #print("quantized_weights is: ", quantized_weights.mean())
            param.mul_(quantized_weights.view(param.data.shape))
            k += t

            # Forward pass after quantization
        output = model_copy(net_input)
        #reg = torch.sigmoid(p).sum()
        # with torch.no_grad():
        #     print(output.shape,noise_var.shape)
        reg = (torch.sigmoid(p) * torch.log((torch.sigmoid(p)+1e-6)/prior) +
               (1-torch.sigmoid(p)) * torch.log((1-torch.sigmoid(p)+1e-6)/(1-prior))).sum()
        # p_mean = torch.sigmoid(p).mean()
        # reg =  (p_mean * torch.log((p_mean+1e-6)/prior) + (1-p_mean) * torch.log((1-p_mean+1e-6)/(1-prior)))/kl
        #reg = torch.abs(torch.sigmoid(p)).sum()

        # reg = torch.abs(torch.sum(torch.sigmoid(p))-sparsity*numel)

        # Compute loss based on the dissimilarity between quantized model output and noisy image
        #quantization_loss = mse(output, noise_var) + kl*reg
        quantization_loss = mse(output,noise_var) + kl*reg
        quantization_loss.backward()
        optimizer_p.step()
        # with torch.no_grad():
        #     print(reg,quantization_loss)


        #quantization_loss_avg = quantization_loss_accum / num_realizationsx
        

        # Update quantization probabilities using gradient descent
        
        with torch.no_grad():
            if epoch % 1000 == 0:
                print("epoch: ", epoch, "quantization_loss: ", quantization_loss.item())
                quant_loss.append(quantization_loss.item()) 
                print("p mean is:",p.mean())
    

        if epoch == num_steps - 1:
            logits_flat = torch.sigmoid(p).view(-1).cpu().detach().numpy()
            all_logits.extend(logits_flat)

    # # Update the actual model based on the quantized weights
    # with torch.no_grad():
    #     if mask_opt=='single':          
    #         out = draw_one_mask(p, model,net_input)
    #     elif mask_opt=='multiple':
    #         print("here")
    #         out = draw_multiple_masks(p, model,net_input)
    #     else:
    #         out = deterministic_rounding(p, model,net_input)


        # k = 0
        # for i, param in enumerate(model.parameters()):
        #     t = len(param.view(-1))
        #     logits = p[:, k:(k+t)].t()
        #     quantized_weights = soft_quantize(torch.sigmoid(logits), q, temperature=0.2)
        #     hard_quant = torch.round(quantized_weights)
        #     print(hard_quant)
        #     param.data = param.data * hard_quant.view(param.data.shape)
        #     k += t
        #     all_quantized_weights.extend(quantized_weights.cpu().numpy().flatten())


    # out = model(net_input)
    # out_np = out.detach().cpu().numpy()[0]
    # img_np = img_var.detach().cpu().numpy()
    # psnr_gt  = compare_psnr(img_np, out_np)
    # print("PSNR of output image is: ", psnr_gt)
    ## save the figure 
    # plt.figure()
    # plt.imshow(out_np[0,:,:])
    # plt.savefig(f'output_{ino}.png')
    
    os.makedirs(f'histogram_centeredl1_{ino}', exist_ok=True) 

    #Plot a histogram for all the quantized weights
    plt.hist(all_logits, bins=50, alpha=0.5, label='All Layers')
    plt.title(f'Distribution of p for sparsity level {sparsity}')
    plt.xlabel('Value of p')
    plt.ylabel('Frequency')
    plt.savefig(f'histogram_centeredl1_{ino}/all_layers_histogram_q_{ino}_{sparsity}_{kl}.png')
    plt.clf()

    return p,quant_loss

def draw_one_mask(logits, model,net_input):
    k = 0
    for i, param in enumerate(model.parameters()):
        t = len(param.view(-1))
        log = logits[:, k:(k+t)].t()
        quantized_weights = soft_quantize(torch.sigmoid(log), q=2, temperature=0.2)
        hard_quant = torch.round(quantized_weights)
        param.data = param.data * quantized_weights.view(param.data.shape)
        k += t
    output = model(net_input)
    return output


def draw_multiple_masks(logits, model,net_input, num_masks=10):
    mask_prob = torch.sigmoid(logits)
    hard_quantized_images = []

    for _ in range(num_masks):
        # masks = torch.bernoulli(mask_prob)
        # hard_quant = torch.round(masks)

        k = 0
        for i, param in enumerate(model.parameters()):
            t = len(param.view(-1))
            log = logits[:, k:(k+t)].t()
            quantized_weights = soft_quantize(torch.sigmoid(log), q=2, temperature=0.2)
            #hard_quant = torch.round(quantized_weights)
            param.data = param.data * quantized_weights.view(param.data.shape)
            k += t
        output = model(net_input)
        hard_quantized_images.append(output)

    average_output = torch.mean(torch.stack(hard_quantized_images), dim=0)
    return average_output

# def deterministic_rounding(logits, model,net_input,sparsity=0.05):
#     #hard_quant = make_mask(logits)
#     hard_quant = make_mask_with_sparsity(logits,sparsity)
#     with torch.no_grad():
#         k = 0
#         for i, param in enumerate(model.parameters()):
#             t = len(param.view(-1))
#             param.data = param.data * hard_quant[k:(k + t)].view(param.data.shape)
#             k += t
#         _ = print_nonzeros(model)   
#     output = model(net_input)
#     return output

def deterministic_rounding(logits, model, net_input, sparsity=0.05):
    hard_quant = make_mask_with_sparsity(logits, sparsity)
    print_nonzeros(model) 
    output = model(net_input)
    #torch.save(hard_quant, 'hard_quant.pt')
    #hard_quant = torch.load('hard_quant.pt')

    # Count the number of 0's and 1's
    # num_zeros = (hard_quant == 0).sum().item()
    # num_ones = (hard_quant == 1).sum().item()

    # # Total number of elements
    # total_elements = hard_quant.numel()

    # # Calculate percentages
    # percent_zeros = (num_zeros / total_elements) * 100
    # percent_ones = (num_ones / total_elements) * 100

    # # Print the results
    # print(f"Number of zeros: {num_zeros} ({percent_zeros:.2f}% of total)")
    # print(f"Number of ones: {num_ones} ({percent_ones:.2f}% of total)")
    # print(f"Total number of elements in hard_quant: {total_elements}")

    #with torch.no_grad():
        # k = 0
        # total_elements = 0
        # for i, param in enumerate(model.parameters()):
        #     t = len(param.view(-1))
        #     before_zeros = (param == 0).sum().item()
        #     mask_part = hard_quant[k:(k + t)]
        #     param.data = param.data * mask_part.view(param.data.shape)
        #     after_zeros = (param == 0).sum().item()
        #     print(f"Layer {i}: Zeros before: {before_zeros}, Zeros after: {after_zeros}")
        #     k += t

        #print(f"Total elements in parameters: {total_elements}, Elements in hard_quant: {hard_quant.numel()}")   
    return output


def make_mask(logits):
    mask_prob = torch.sigmoid(logits)
    hard_quant = torch.round(mask_prob.t())
    return hard_quant

# def make_mask_with_sparsity(logits,sparsity=0.05):
#     mask_prob = torch.sigmoid(logits)
#     num_elements = logits.numel()
#     num_to_keep = int(sparsity * num_elements)  
#     threshold = torch.topk(logits.view(-1), num_to_keep, largest=True).values.min()
#     sparse_mask = (logits >= threshold).float()
#     sparse_mask_prob = mask_pr
# ob * sparse_mask
#     hard_quant = torch.round(sparse_mask_prob.t())
#     return hard_quant

def make_mask_with_sparsity(logits, sparsity=0.05):
    mask_prob = torch.sigmoid(logits)
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
    num_to_randomly_select = max(0, num_to_keep - (values > threshold).sum().item())
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



def add_noise(model, noise_scale,lr):
    for n in [x for x in model.parameters() if len(x.size()) == 4]:
        ran = torch.rand(n.shape)
        noise_tensor = ran * noise_scale * lr  # Renamed variable
        noise_tensor = noise_tensor.type(dtype)
        n.data = n.data + noise_tensor



def train_sparse(masked_model, net_input, mask, img_var, noise_var, learning_rate=0.01, max_step=40000,show_every=200, lr_step=100000, lr_gamma=0.1, device='cuda:0'):
    # Setting the device for the model and tensors
    masked_model = masked_model.to(device)
    img_var = np_to_torch(img_var).type(dtype).to(device)
    noise_var = np_to_torch(noise_var).type(dtype).to(device)
    net_input = net_input.to(device)
    mask = mask.to(device)
    psrn_noisy_last=0.0

    mse = torch.nn.MSELoss().type(dtype)
    psnr_lists = []
    print_nonzeros(masked_model)
    # Define the optimizer
    optimizer = torch.optim.Adam(masked_model.parameters(), lr=learning_rate)

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # Forward pass to compute initial PSNR
    with torch.no_grad():
        initial_out = masked_model(net_input)
        # initial_psnr = compare_psnr(img_var.detach().cpu().numpy(), initial_out.detach().cpu().numpy())
        # print("Initial PSNR of output image is: ", initial_psnr)
    #noise_scale=5e-3
    #start_time = time.time()
    for epoch in range(max_step):
        
        # start_time = time.time()
        optimizer.zero_grad()
        out = masked_model(net_input)

        total_loss = mse(out, noise_var)
        with torch.no_grad():
            if epoch==0:
                print("total_loss is:",total_loss)
        total_loss.backward()
        # end_time = time.time()
        # duration = end_time - start_time
        # print("duration is:",duration)



        # Adjust gradients according to the mask
        k = 0
        for param in masked_model.parameters():
            t = len(param.view(-1))
            param.grad.data = param.grad.data * mask[k:(k+t)].view(param.grad.data.shape)
            k += t

        optimizer.step()

        #print("duration is:",duration)
        #scheduler.step()
       # add_noise(masked_model, noise_scale,learning_rate)
        #mask_network(mask,masked_model)

        if epoch % show_every == 0:
            # Calculating PSNR
            out_np = out.detach().cpu().numpy()
            img_np = img_var.detach().cpu().numpy()
            psnr_gt = compare_psnr(img_np, out_np)
            psnr_noisy = compare_psnr(noise_var.detach().cpu().numpy(), out_np)
            psnr_lists.append(psnr_gt)

            # if psnr_noisy - psrn_noisy_last < -2: 
            #     print('Falling back to previous checkpoint.')
            #     for new_param, net_param in zip(last_net, masked_model.parameters()):
            #         net_param.detach().copy_(new_param.cuda())
            # else:
            #     last_net = [x.detach().cpu() for x in masked_model.parameters()]
            #     psrn_noisy_last = psnr_noisy

            #print_nonzeros(masked_model)
            # ## imshow the figure and save the figure (make sure you transppose it properly)
            # plt.figure()
            # plt.imshow(out_np.transpose(1,2,0))
            # plt.savefig(f'gif_files/output_{epoch}.png')
            


            # Print epoch, loss and PSNR
            print("epoch: ", epoch, "loss: ", total_loss.item(), "PSNR: ", psnr_gt, "PSNR_noisy: ", psnr_noisy)
    # end_time = time.time()
    # duration = end_time - start_time
    # print("duration is:",duration)
    return psnr_lists, out_np


def train_dense(net, net_input,  img_var, noise_var,learning_rate=1e-3, max_step=40000, show_every=1000,lr_step=100000, lr_gamma=0.1, device='cuda:0'):
    # Setting the device for the model and tensors
    net= net.to(device)
    img_var = np_to_torch(img_var).type(dtype).to(device)
    noise_var = np_to_torch(noise_var).type(dtype).to(device)
    net_input = net_input.to(device)

    mse = torch.nn.MSELoss().type(dtype)
    psnr_lists = []
    print_nonzeros(net)
    # Define the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # Forward pass to compute initial PSNR
    with torch.no_grad():
        initial_out = net(net_input)
        initial_psnr = compare_psnr(img_var.detach().cpu().numpy(), initial_out.detach().cpu().numpy()[0])
        print("Initial PSNR of output image is: ", initial_psnr)
    
    for epoch in range(max_step):
        optimizer.zero_grad()
        out = net(net_input)
        total_loss = mse(out, noise_var)
        total_loss.backward()

        optimizer.step()
        scheduler.step()

        if epoch % show_every == 0:
            # Calculating PSNR
            out_np = out.detach().cpu().numpy()[0]
            img_np = img_var.detach().cpu().numpy()
            psnr_gt = compare_psnr(img_np, out_np)
            psnr_lists.append(psnr_gt)
            # print_nonzeros(masked_model)

            # Print epoch, loss and PSNR
            print("epoch: ", epoch, "loss: ", total_loss.item(), "PSNR: ", psnr_gt)

    return psnr_lists, out_np    

def train_deep_decoder(k,img_var, noise_var,learning_rate=0.01, max_step=40000, show_every=1000,device='cuda:1'):
    output_depth = img_var.shape[0]
    print(img_var.shape)
    img_var = np_to_torch(img_var).type(dtype).to(device)
    noise_var = np_to_torch(noise_var).type(dtype).to(device)
    mse = torch.nn.MSELoss().type(dtype)
    num_channels = [128]*k
    totalupsample = 2**len(num_channels)
    print(totalupsample,img_var.shape)
    width = int(img_var.shape[2]/totalupsample)
    height = int(img_var.shape[2]/totalupsample)
    print(width,height)
    shape = [1,num_channels[0], width, height]  
    net_input = Variable(torch.zeros(shape))
    net_input.data.uniform_()
    net_input.data *= 1./10
    net_input = net_input.type(dtype)
    net = decodernw(output_depth, num_channels_up=num_channels , upsample_first=True).type(dtype)    
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
            out_np = out.detach().cpu().numpy()[0]
            img_np = img_var.detach().cpu().numpy()
            psnr_gt = compare_psnr(img_np, out_np)
            psnr_lists.append(psnr_gt)
            # print_nonzeros(masked_model)
            # Print epoch, loss and PSNR
            print("epoch: ", epoch, "loss: ", total_loss.item(), "PSNR: ", psnr_gt)    
    return psnr_lists, out_np   



def print_nonzeros(model):
    nonzero = total = 0
    for name, param in model.named_parameters():
        nz_count = torch.count_nonzero(param.data)
        total_params = param.data.numel() 
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} \
            ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {param.data.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')
    return (round((nonzero.item()/total)*100,1))



def train_sparse_inpaint(masked_model, net_input, mask, img_var, mask_var,img_noise, learning_rate=0.01, max_step=40000,show_every=1000, lr_step=100000, lr_gamma=0.1, device='cuda:0'):
    # Setting the device for the model and tensors
    masked_model = masked_model.to(device)
    img_var = np_to_torch(img_var).type(dtype).to(device)
    img_noise = np_to_torch(img_noise).type(dtype).to(device)
    net_input = net_input.to(device)
    mask = mask.to(device)
    mask_var = np_to_torch(mask_var).type(dtype).to(device)    

    mse = torch.nn.MSELoss().type(dtype)
    psnr_lists = []
    print_nonzeros(masked_model)
    # Define the optimizer
    optimizer = torch.optim.Adam(masked_model.parameters(), lr=learning_rate)

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    # Forward pass to compute initial PSNR
    with torch.no_grad():
        initial_out = masked_model(net_input)
        print(initial_out.shape,img_var.shape)
        initial_psnr = compare_psnr(img_var.detach().cpu().numpy(), initial_out.detach().cpu().numpy())
        print("Initial PSNR of output image is: ", initial_psnr)
    #noise_scale=5e-3
    for epoch in range(max_step):
        optimizer.zero_grad()
        out = masked_model(net_input)
        total_loss = mse(out*mask_var, img_noise*mask_var)
        total_loss.backward()


        # Adjust gradients according to the mask
        k = 0
        for param in masked_model.parameters():
            t = len(param.view(-1))
            param.grad.data = param.grad.data * mask[k:(k+t)].view(param.grad.data.shape)
            k += t

        optimizer.step()
        scheduler.step()
       # add_noise(masked_model, noise_scale,learning_rate)
        #mask_network(mask,masked_model)

        if epoch % show_every == 0:
            # Calculating PSNR
            out_np = out.detach().cpu().numpy()
            img_np = img_var.detach().cpu().numpy()
            psnr_gt = compare_psnr(img_np, out_np)
            psnr_lists.append(psnr_gt)
            # print_nonzeros(masked_model)
            # ## imshow the figure and save the figure (make sure you transppose it properly)
            # plt.figure()
            # plt.imshow(out_np.transpose(1,2,0))
            # plt.savefig(f'gif_files/output_{epoch}.png')
            


            # Print epoch, loss and PSNR
            print("epoch: ", epoch, "loss: ", total_loss.item(), "PSNR: ", psnr_gt)

    return psnr_lists, out_np
    


def learn_quantization_probabilities_dip_inpaint(model,net_input,img_var,mask_var, num_steps, lr,ino, q=2,kl=1e-9,prior_sigma=0.0):
    device = next(model.parameters()).device
    mse = torch.nn.MSELoss().type(dtype)
    img_var = np_to_torch(img_var).type(dtype)
    #noise_var = np_to_torch(noise_var).type(dtype)
    # Initialize quantization probabilities (p) and make sure they require gradients
    _, p, _ = quant_initialization(model, 1.0, q)
    p.requires_grad_(True)
    optimizer_p = torch.optim.Adam([p], lr=lr)
    num_realizations =5
    prior=sigmoid(prior_sigma)

    all_quantized_weights = []
    quant_loss =[]

    for epoch in range(num_steps):
        
        model_copy = copy.deepcopy(model)
        for param in model_copy.parameters():
            param.requires_grad = False
        quantization_loss_accum = 0.0

#        for realization in range(num_realizations):
        optimizer_p.zero_grad()
        k = 0
        for i, param in enumerate(model_copy.parameters()):
            t = len(param.view(-1))
            logits = p[:, k:(k+t)].t()
            quantized_weights = soft_quantize(torch.sigmoid(logits), q, temperature=0.2)
            #print("quantized_weights is: ", quantized_weights.mean())
            param.mul_(quantized_weights.view(param.data.shape))
            k += t

            # Forward pass after quantization
        output = model_copy(net_input)
        #reg = torch.sigmoid(p).sum()
        # with torch.no_grad():
        #     print(output.shape,noise_var.shape)
        reg=  (torch.sigmoid(p) * torch.log((torch.sigmoid(p)+1e-6)/prior) + (1-torch.sigmoid(p)) * torch.log((1-torch.sigmoid(p)+1e-6)/(1-prior))).sum()

            # Compute loss based on the dissimilarity between quantized model output and noisy image
        quantization_loss = mse(output*mask_var, img_var*mask_var) + kl*reg
        quantization_loss.backward()
        optimizer_p.step()
        
        with torch.no_grad():
            if epoch % 1000 == 0:
                print("epoch: ", epoch, "quantization_loss: ", quantization_loss.item())
                quant_loss.append(quantization_loss.item()) 
                print("p mean is:",p.mean())


    return p,quant_loss


def generate_specific_quarter_chessboard(img_shape, square_size, noise_level, quarter):
    """
    Generate a chessboard pattern with added noise, keeping only a specific quarter.

    Parameters:
    img_shape (tuple): Shape of the image (channels, height, width).
    square_size (int): Size of each square in the checkerboard.
    noise_level (float): Standard deviation of the Gaussian noise added to the image.
    quarter (int): The quarter of the chessboard to keep (1, 2, 3, or 4).

    Returns:
    np.array: Chessboard pattern with noise, with only the specified quarter present.
    """
    # Create the chessboard pattern
    rows, cols = img_shape[1] // square_size, img_shape[2] // square_size
    chessboard = np.kron([[1, 0] * cols, [0, 1] * cols] * rows, np.ones((square_size, square_size)))
    chessboard = np.tile(chessboard[:img_shape[1], :img_shape[2]], (img_shape[0], 1, 1))

    # Add Gaussian noise
    noise = np.random.normal(scale=noise_level, size=img_shape)
    noisy_chessboard = chessboard + noise

    # Masking to keep only the specified quarter
    half_height = img_shape[1] // 2
    half_width = img_shape[2] // 2
    mask = np.zeros_like(noisy_chessboard)
    if quarter == 1:
        mask[:, :half_height, :half_width] = 1
    elif quarter == 2:
        mask[:, :half_height, half_width:] = 1
    elif quarter == 3:
        mask[:, half_height:, :half_width] = 1
    elif quarter == 4:
        mask[:, half_height:, half_width:] = 1
    else:
        raise ValueError("Invalid quarter. Choose 1, 2, 3, or 4.")

    return noisy_chessboard * mask    


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def weights_init(m, scale=1.0):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.mul_(scale)
        if m.bias is not None:
            m.bias.data.mul_(scale)            



def learn_quantization_probabilities_dip_deblur(model,net_input,img_var,noise_var, num_steps, lr,ino, q=3,kl=1e-5,prior_sigma=0.0,sparsity=0.5):
    device = next(model.parameters()).device
    mse = torch.nn.MSELoss().type(dtype)
    img_var = np_to_torch(img_var).type(dtype)
    noise_var = np_to_torch(noise_var).type(dtype)
    sigma=2.0
    # Initialize quantization probabilities (p) and make sure they require gradients
    _, p, _ = quant_initialization(model, 1.0, q)
    numel=p.numel()
    p.requires_grad_(True)
    optimizer_p = torch.optim.Adam([p], lr=lr)
    num_realizations =5
    prior=sigmoid(prior_sigma)

    all_logits = []
    quant_loss =[]

    for epoch in range(num_steps):
        
        model_copy = copy.deepcopy(model)
        for param in model_copy.parameters():
            param.requires_grad = False
        quantization_loss_accum = 0.0

#        for realization in range(num_realizations):
        optimizer_p.zero_grad()
        k = 0
        for i, param in enumerate(model_copy.parameters()):
            t = len(param.view(-1))
            logits = p[:, k:(k+t)].t()
            quantized_weights = soft_quantize(torch.sigmoid(logits), q, temperature=0.2)
            #print("quantized_weights is: ", quantized_weights.mean())
            param.mul_(quantized_weights.view(param.data.shape))
            k += t

            # Forward pass after quantization
        output = model_copy(net_input)
        blurred_out = convolve_with_gaussian_torch(output, sigma)
        #reg = torch.sigmoid(p).sum()
        # with torch.no_grad():
        #     print(output.shape,noise_var.shape)
        reg =  (torch.sigmoid(p) * torch.log((torch.sigmoid(p)+1e-6)/prior) + (1-torch.sigmoid(p)) * torch.log((1-torch.sigmoid(p)+1e-6)/(1-prior))).sum()
        #reg = torch.abs(torch.sigmoid(p)).sum()
        #reg = torch.abs(torch.sum(torch.sigmoid(p))-sparsity*numel)

            # Compute loss based on the dissimilarity between quantized model output and noisy image
        #quantization_loss = mse(output, noise_var) + kl*reg
        quantization_loss = mse(output,noise_var) + kl*reg
        quantization_loss.backward()
        optimizer_p.step()
        # with torch.no_grad():
        #     print(reg,quantization_loss)


        #quantization_loss_avg = quantization_loss_accum / num_realizationsx
        

        # Update quantization probabilities using gradient descent
        
        with torch.no_grad():
            if epoch % 1000 == 0:
                print("epoch: ", epoch, "quantization_loss: ", quantization_loss.item())
                quant_loss.append(quantization_loss.item()) 
                print("p mean is:",p.mean())
    

        if epoch == num_steps - 1:
            logits_flat = torch.sigmoid(p).view(-1).cpu().detach().numpy()
            all_logits.extend(logits_flat)

    # # Update the actual model based on the quantized weights
    # with torch.no_grad():
    #     if mask_opt=='single':          
    #         out = draw_one_mask(p, model,net_input)
    #     elif mask_opt=='multiple':
    #         print("here")
    #         out = draw_multiple_masks(p, model,net_input)
    #     else:
    #         out = deterministic_rounding(p, model,net_input)


        # k = 0
        # for i, param in enumerate(model.parameters()):
        #     t = len(param.view(-1))
        #     logits = p[:, k:(k+t)].t()
        #     quantized_weights = soft_quantize(torch.sigmoid(logits), q, temperature=0.2)
        #     hard_quant = torch.round(quantized_weights)
        #     print(hard_quant)
        #     param.data = param.data * hard_quant.view(param.data.shape)
        #     k += t
        #     all_quantized_weights.extend(quantized_weights.cpu().numpy().flatten())


    # out = model(net_input)
    # out_np = out.detach().cpu().numpy()[0]
    # img_np = img_var.detach().cpu().numpy()
    # psnr_gt  = compare_psnr(img_np, out_np)
    # print("PSNR of output image is: ", psnr_gt)
    ## save the figure 
    # plt.figure()
    # plt.imshow(out_np[0,:,:])
    # plt.savefig(f'output_{ino}.png')
    
    # os.makedirs(f'histogram_l1_{ino}', exist_ok=True) 

    # #Plot a histogram for all the quantized weights
    # plt.hist(all_logits, bins=50, alpha=0.5, label='All Layers')
    # plt.title(f'Distribution of p for sparsity level {sparsity}')
    # plt.xlabel('Value of p')
    # plt.ylabel('Frequency')
    # plt.savefig(f'histogram_l1_{ino}/all_layers_histogram_q_{ino}_{sparsity}_{kl}.png')
    # plt.clf()

    return p,quant_loss


def gaussian_kernel(size, sigma):
    """
    Generates a 2D Gaussian kernel.
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def convolve_with_gaussian(image_np, sigma):
    """
    Convolves a 3-channel image with a Gaussian kernel.
    Ensures the output image has the same size as the input.
    """
    if image_np.ndim == 3 and image_np.shape[0] == 3:  # Check if image has 3 channels
        blurred_image = np.zeros_like(image_np)
        kernel_size = int(sigma * 3) * 2 + 1  # Kernel size
        for c in range(3):
            blurred_image[c,...] = gaussian_filter(image_np[c,...], sigma=sigma, mode='reflect', truncate=3.0)
    else:
        kernel_size = int(sigma * 3) * 2 + 1
        blurred_image = gaussian_filter(image_np, sigma=sigma, mode='reflect', truncate=3.0)
    return blurred_image

def gaussian_kernel_torch(kernel_size, sigma):
    """
    Generates a 2D Gaussian kernel using PyTorch.
    """
    ax = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, kernel_size)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel

def convolve_with_gaussian_torch(image_tensor, sigma):
    """
    Convolves an image tensor with a Gaussian kernel using PyTorch.
    Assumes the image tensor is in BCHW format.
    """
    batch_size, channels, _, _ = image_tensor.shape
    kernel_size = int(sigma * 3) * 2 + 1
    kernel = gaussian_kernel_torch(kernel_size, sigma).to(image_tensor.device)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
    
    # Ensure the kernel is [out_channels, in_channels/groups, height, width]
    padding = kernel_size // 2
    blurred_image = F.conv2d(image_tensor, kernel, padding=padding, groups=channels)
    
    return blurred_image