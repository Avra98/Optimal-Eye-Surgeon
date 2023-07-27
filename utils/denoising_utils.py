import os
from .common_utils import *
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
import matplotlib.pylab as plt 

import torch
import torch.optim
import time
#from skimage.measure import compare_psnr
import _pickle as cPickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

        
def get_noisy_image(img_np, sigma):
    """Adds Gaussian noise to an image.
    
    Args: 
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)
    img_noisy_pil = np_to_pil(img_noisy_np)

    return img_noisy_pil, img_noisy_np


def ind_loss(network: nn.Module,net_input,img_var,noise_var):
    mse = torch.nn.MSELoss().type(dtype)
    img_var = np_to_torch(img_var).type(dtype)
    noise_var = np_to_torch(noise_var).type(dtype)
    out = network(net_input)
    total_loss = mse(out, noise_var) 
    return total_loss

def compute_hvp(network: nn.Module, ind_loss, net_input_list, img_np_list, img_noise_np_list, 
                vector: Tensor):
    """Compute a Hessian-vector product."""
    p = len(parameters_to_vector(network.parameters()))
    hvp = torch.zeros(p, dtype=torch.float, device='cuda')
    vector = vector.cuda()
    for i in range(len(img_np_list)): 
        loss = ind_loss(network,net_input_list[i],img_np_list[i],img_noise_np_list[i])/len(img_np_list)
        grads = torch.autograd.grad(loss, inputs=network.parameters(), create_graph=True)
        dot = parameters_to_vector(grads).mul(vector).sum()
        grads = [g.contiguous() for g in torch.autograd.grad(dot, network.parameters(), retain_graph=True)]
        hvp += parameters_to_vector(grads)
    return hvp

def get_hessian_eigenvalues(network: nn.Module, ind_loss, net_input_list, img_np_list, img_noise_np_list, 
                            neigs=6):
    """ Compute the leading Hessian eigenvalues. """
    hvp_delta = lambda delta: compute_hvp(network, ind_loss, net_input_list, img_np_list, img_noise_np_list,delta).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    evals, evecs = lanczos(hvp_delta, nparams, neigs=neigs)
    return evals


def get_jac_norm(network: nn.Module, net_input_list,n_iters=50):
    dy_dw=0.0
    for i in range(len(net_input_list)):
        out = network(net_input_list[i])
        for _ in range(n_iters):
            dot = 0.0
            v = torch.randn(out.shape, requires_grad=False).cuda()
            dot = out.mul(v).sum()/(v.shape[2]**2)

            grads = torch.autograd.grad(dot, inputs=network.parameters(), retain_graph=True)
            norm_square_sum = sum(torch.norm(g)**2 for g in grads) 
            dy_dw +=norm_square_sum/n_iters
    return dy_dw 


def get_trace(network: nn.Module, ind_loss, net_input_list, img_np_list, img_noise_np_list,n_iters=50):
    hvp_delta = lambda delta: compute_hvp(network, ind_loss, net_input_list, img_np_list, img_noise_np_list,delta).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))
    trace=0.0
    for _ in range(n_iters):
        v = torch.randn(nparams)
        Hv = hvp_delta(v)
        trace += torch.dot(Hv, v).item()
    return trace/n_iters 
        
    

def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        gpu_vec = torch.tensor(vec, dtype=torch.float).cuda()
        return matrix_vector(gpu_vec)

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()




                            

def get_hessian_spectrum(network: nn.Module, ind_loss, net_input_list, img_np_list, img_noise_np_list,  
                         iter: int = 100, n_v: int = 1):
    """ Compute the Hessian eigenspectrum using the stochastic Lanczos Quadrature (SLQ). """

    hvp_delta = lambda delta: compute_hvp(network, ind_loss, net_input_list, img_np_list, img_noise_np_list,delta).detach().cpu()
    nparams = len(parameters_to_vector((network.parameters())))

    eigen_list_full = []
    weight_list_full = []

    for k in range(n_v):
        v = torch.randn(nparams).normal_().to('cuda') # Generate a normal random vector
        v = v / torch.norm(v) # normalize the vector

        # Standard lanczos algorithm initialization
        v_list = [v]
        w_list = []
        alpha_list = []
        beta_list = []

        for i in range(iter):
            w_prime = hvp_delta(v_list[-1]).to('cuda')
            #print(i)
            
            
            if i == 0:
                alpha = torch.dot(w_prime, v)
                alpha_list.append(alpha.item())
                w = w_prime - alpha * v
                w_list.append(w)
            else:
                beta = torch.norm(w).to('cuda')
                beta_list.append(beta.item())
                if beta != 0.:
                    v = w / beta
                    v_list.append(v)
                else:
                    v = torch.randn(nparams).normal_().to('cuda')
                    v_list.append(v / torch.norm(v))
                w_prime = hvp_delta(v_list[-1]).to('cuda')
                alpha = torch.dot(w_prime, v)
                alpha_list.append(alpha.item())
                w = w_prime - alpha * v - beta * v_list[-2]

        T = torch.zeros(iter, iter).to('cuda')
        for i in range(len(alpha_list)):
            T[i, i] = alpha_list[i]
            if i < len(alpha_list) - 1:
                T[i + 1, i] = beta_list[i]
                T[i, i + 1] = beta_list[i]
        
        eigenvalues, eigenvectors = torch.linalg.eig(T)
        eigen_list_full.append(eigenvalues.cpu().numpy())
        weight_list_full.append((eigenvectors[0]**2).cpu().numpy())

    return eigen_list_full, weight_list_full




def get_esd_plot(eigenvalues, weights, filename='esd_plot.png'):
    density, grids = density_generate(eigenvalues, weights)
    plt.semilogy(grids, density + 1.0e-7)
    plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
    plt.xlabel('Eigenvalue', fontsize=14, labelpad=10)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    #plt.close()



def density_generate(eigenvalues,
                     weights,
                     num_bins=10000,
                     sigma_squared=1e-5,
                     overhead=0.01):

    eigenvalues = np.array(eigenvalues)
    weights = np.array(weights)

    lambda_max = np.mean(np.max(eigenvalues, axis=1), axis=0) + overhead
    lambda_min = np.mean(np.min(eigenvalues, axis=1), axis=0) - overhead

    grids = np.linspace(lambda_min, lambda_max, num=num_bins)
    sigma = sigma_squared * max(1, (lambda_max - lambda_min))

    num_runs = eigenvalues.shape[0]
    density_output = np.zeros((num_runs, num_bins))

    for i in range(num_runs):
        for j in range(num_bins):
            x = grids[j]
            tmp_result = gaussian(eigenvalues[i, :], x, sigma)
            density_output[i, j] = np.sum(tmp_result * weights[i, :])
    density = np.mean(density_output, axis=0)
    normalization = np.sum(density) * (grids[1] - grids[0])
    density = density / normalization
    return density, grids


def gaussian(x, x0, sigma_squared):
    return np.exp(-(x0 - x)**2 /
                  (2.0 * sigma_squared)) / np.sqrt(2 * np.pi * sigma_squared)

