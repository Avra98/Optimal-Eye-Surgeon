import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt

# Define file path
file_path1 = 'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/6layers/l1/p_0.pkl'

# Limits for the histogram
lower_limit = 0.02
upper_limit = 0.07

all_logits = []  # List to store all logits

# Load data from .pkl file
with open(file_path1, 'rb') as file:
    data = pickle.load(file)

# Check if data is a tensor and process accordingly
if isinstance(data, torch.Tensor):
    
    logits = torch.sigmoid(data).view(-1)
    logits_flat = logits[(logits >= lower_limit) & (logits <= upper_limit)].cpu().detach().numpy()
    all_logits.extend(logits_flat)
# Assuming non-tensor data is already in the desired form (or add handling here)

# Calculate histogram
counts, bin_edges = np.histogram(all_logits, bins=50, range=(lower_limit, upper_limit))

# Find the values with the largest numbers of repetitions
sorted_indices = np.argsort(counts)[::-1]  # Get indices of sorted counts in descending order
top_counts = counts[sorted_indices][:10]  # Get top 4 counts
top_values = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in sorted_indices[:10]]  # Get top 4 values

# Print the top 4 values and their counts
for i, (value, count) in enumerate(zip(top_values, top_counts), start=1):
    print(f"{i}th most common value: {value}, Count: {count}")

# Plot histogram of the data within the specified limits
plt.figure()
plt.hist(all_logits, bins=50, range=(lower_limit, upper_limit))
plt.title('Histogram of Sigmoid Data within Limits')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Save the figure
plt.savefig('sigmoid_histogram_kl.png')
