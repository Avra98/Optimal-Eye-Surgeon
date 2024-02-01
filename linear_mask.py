import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F


num_samples = 100
num_features = 200

# Generate orthogonal sinusoidal columns (Fourier basis)
t = np.linspace(0, np.pi, num_samples, endpoint=False)  # Interval [0, 2π)
sinusoids = np.array([np.sin((f+1) * t) for f in range(num_samples)]).T

# Ensure J is full row rank by repeating the sinusoidal columns
J = np.hstack((sinusoids, sinusoids))

# # Plotting J
# plt.figure(figsize=(10, 8))
# plt.imshow(J, aspect='auto')
# plt.colorbar()
# plt.title("Matrix J Visualization - Fourier Basis")
# plt.xlabel("Features")
# plt.ylabel("Samples")

# # Save the image
# image_path_fourier_corrected = "matrix_j_fourier_corrected.png"
# plt.savefig(image_path_fourier_corrected)


# Initialize c as a zero vector
c = np.zeros(num_features)

# Choose non-zero values for c
# Larger coefficients for the first 3 columns (low frequencies)
c[:3] = np.random.randn(3) * 10  # Scaling by 10 for larger values

# Smaller coefficients for columns 93-100 (high frequencies)
high_freq_indices = np.random.choice(range(80, 100), 20, replace=False)
c[high_freq_indices] = np.random.randn(20)  # Default scale for smaller values
print(c)

# Compute the ground truth signal y
y = np.dot(J, c)

##plot y and save the plot
plt.plot(y)
plt.savefig('y.png')

# Compute the Moore-Penrose pseudoinverse of J
J_pinv = np.linalg.pinv(J)

# Compute the minimum L2 norm solution c_hat
c_hat = np.dot(J_pinv, y)

c_hat

# Parameters for optimization
num_iterations = 1000
learning_rate = 0.01

# Initialize c with random values
c = torch.randn(num_features, requires_grad=False)

def gumbel_softmax_multi(logits, temperature=0.2):
    logits = torch.cat((logits, 1 - logits.sum(dim=1, keepdim=True).clamp(min=1e-20, max=1-1e-20)), dim=1) 
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-20, max=1-1e-20))).requires_grad_(False)
    y = (torch.log(logits.clamp(min=1e-20, max=1-1e-20)) + gumbel_noise) / temperature
    softmax_output = F.softmax(y, dim=-1)
    return softmax_output

def soft_quantize(logits, q, temperature=0.2):
    soft_samples = gumbel_softmax_multi(logits, temperature)
    with torch.no_grad():
        quant_values = [torch.tensor([1 - i / (q - 1)]) for i in range(q - 1)] + [torch.tensor([0.0])]
        quant_values = torch.cat(quant_values).to(logits.device)
    quantized_output = torch.sum(soft_samples * quant_values, dim=-1)
    return quantized_output

q = 2  # Number of quantization levels
p = torch.nn.Parameter(torch.ones([q-1, num_features], requires_grad=True))

# Optimization setup
learning_rate = 1e-2
optimizer = torch.optim.SGD([p], lr=learning_rate)
num_iterations = 100000
prior=0.02

# Optimization loop to learn the mask probabilities
for iteration in range(num_iterations):
    optimizer.zero_grad()
    # Generate the mask using the soft quantization function
    logits = torch.sigmoid(p)
    #print(logits.T.shape)
    mask = soft_quantize(logits.T, q, temperature=0.2)   
    # Apply the mask to c (for demonstration purposes)
    masked_c = c * mask
    # Example loss function (modify as needed)
    predicted_signal = torch.matmul(torch.Tensor(J), masked_c)
    reg=  (torch.sigmoid(p) * torch.log((torch.sigmoid(p)+1e-6)/prior) + (1-torch.sigmoid(p)) * torch.log((1-torch.sigmoid(p)+1e-6)/(1-prior))).sum()
    loss = torch.mean((torch.Tensor(y) - predicted_signal)**2) +1e-2*reg
    

    # Backpropagate and update the probabilities vector p
    loss.backward()
    optimizer.step()

    if iteration % 1000 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}")


mask=torch.round(torch.sigmoid(p))
print(mask)

