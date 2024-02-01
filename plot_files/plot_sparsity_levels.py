import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ino=5

# Paths to the four .npz files
path1 = f'data/denoising/Set14/mask/{ino}/sparsity/det/0.05/1e-09/out_sparsenet/0.1/psnr_{ino}.npz'
path2 = f'data/denoising/Set14/mask/{ino}/sparsity/det/0.5/1e-09/out_sparsenet/0.1/psnr_{ino}.npz'
path3 = f'data/denoising/Set14/mask/{ino}/sparsity/det/0.8/1e-09/out_sparsenet/0.1/psnr_{ino}.npz'
#path4 = 'data/denoising/Dataset/mask/3/unet/det/10.0/1e-09/out_sparsenet/0.1/psnr_3.npz'
path5 = f'data/denoising/Set14/mask/{ino}/ADAM(sigma=0.1,lr=0.01,decay=0,beta=0,reg=0.05)/deepdecoder/psnr_{ino}.npz'
#path6 = f'data/denoising/Set14/mask/{ino}/ADAM(sigma=0.1,lr=0.001,decay=0,beta=0,reg=0.05)/psnr_{ino}.npz'
path6 = f'data/denoising/Set14/mask/{ino}/vanilla/psnr_{ino}.npz'

# Loading data from the npz files
data1 = np.load(path1)
data2 = np.load(path2)
data3 = np.load(path3)
#data4 = np.load(path4)
data5 = np.load(path5)
data6 = np.load(path6)

# Extracting the 'psnr' variable from each file
psnr1 = data1['psnr']
psnr2 = data2['psnr']
psnr3 = data3['psnr']
#psnr4 = data4['psnr']
psnr5 = data5['psnr']
psnr6 = data6['psnr']

print(psnr5.shape)
max_length = max(len(psnr1), len(psnr2), len(psnr3), len(psnr5),len(psnr6))
epochs = np.arange(max_length) * 200 
#epochs5 = np.arange(len(psnr5)) * 1000 

# Function to extend an array if needed
def extend_array(arr, length):
    if len(arr) < length:
        last_val = arr[-1]
        extended_arr = np.append(arr, [last_val] * (length - len(arr)))
        return extended_arr
    else:
        return arr

# Extending the PSNR arrays
psnr1_extended = extend_array(psnr1, max_length)
psnr2_extended = extend_array(psnr2, max_length)
psnr3_extended = extend_array(psnr3, max_length)
#psnr4_extended = extend_array(psnr4, max_length)    
psnr5_extended = extend_array(psnr5, max_length)
psnr6_extended = extend_array(psnr6, max_length)
# Plotting the data
plt.figure(figsize=(10, 6))
# plt.plot(epochs, psnr1_extended, label=' -0.1 ')
plt.plot(epochs, psnr1_extended, label='3% Sparse-DIP ')
plt.plot(epochs, psnr2_extended, label='50% Sparse-DIP')
plt.plot(epochs, psnr3_extended, label='80% Sparse-DIP')
plt.plot(epochs, psnr5_extended, label='Deep Decoder')
plt.plot(epochs, psnr6_extended, label='Vanilla DIP')
# Enlarging the x-axis, y-axis, and their ticks
plt.xlabel('Number of Epochs', fontsize=14)
plt.ylabel('PSNR', fontsize=14)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.ylim(22, 31)


plt.title('PSNR vs. Number of Epochs', fontsize=16)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True)
plt.savefig(f'data/denoising/Set14/mask/{ino}/psnr_comb{ino}.svg')
plt.savefig(f'data/denoising/Set14/mask/{ino}/psnr_comb{ino}.png')