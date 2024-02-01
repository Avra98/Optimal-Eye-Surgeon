import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

## figure -0  from set-14 datset 
# # Paths to the four .npz files
# path1 = 'data/denoising/Set14/mask/0/SAM(sigma=0.1,lr=0.01,decay=0,beta=0)/det/-1.8/1e-09/out_sparsenet/0.1/psnr_0.npz'
# path2 = 'data/denoising/Set14/mask/0/ADAM(sigma=0.1,lr=0.01,decay=0,beta=0,reg=0.05)/deepdecoder/0.1/psnr_0.npz'
# path3 = 'data/denoising/Set14/mask/0/sgld/psnr_0.npz'
# path4 = 'data/denoising/Set14/mask/0/ADAM(sigma=0.1,lr=0.001,decay=0,beta=0,reg=0.05)/psnr_0.npz'
# path5 = 'data/denoising/Set14/mask/0/pat/14_0.2/trans_0_sparsenet/0.1/psnr_0.npz'

# # Loading data from the npz files
# data1 = np.load(path1)
# data2 = np.load(path2)
# data3 = np.load(path3)
# data4 = np.load(path4)
# data5 = np.load(path5)

# # Extracting the 'psnr' variable from each file
# psnr1 = data1['psnr']
# psnr2 = data2['psnr']
# psnr3 = data3['psnr']
# psnr4 = data4['psnr']
# psnr5 = data5['psnr']

# max_length = max(len(psnr1), len(psnr2), len(psnr3), len(psnr4), len(psnr5))
# epochs = np.arange(max_length) * 200 

# # Function to extend an array if needed
# def extend_array(arr, length):
#     if len(arr) < length:
#         last_val = arr[-1]
#         extended_arr = np.append(arr, [last_val] * (length - len(arr)))
#         return extended_arr
#     else:
#         return arr

# # Extending the PSNR arrays
# psnr1_extended = extend_array(psnr1, max_length)
# psnr2_extended = extend_array(psnr2, max_length)
# psnr3_extended = extend_array(psnr3, max_length)
# psnr4_extended = extend_array(psnr4, max_length)    
# psnr5_extended = extend_array(psnr5, max_length)

# # Plotting the data
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, psnr1_extended, label='Sparse-DIP')
# plt.plot(epochs, psnr2_extended, label='Deep Decoder')
# plt.plot(epochs, psnr3_extended, label='SGLD')
# plt.plot(epochs, psnr4_extended, label='Dense DIP')
# plt.plot(epochs, psnr5_extended, label='IMP')

# # Enlarging the x-axis, y-axis, and their ticks
# plt.xlabel('Number of Epochs', fontsize=14)
# plt.ylabel('PSNR', fontsize=14)
# plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylim(18, 30)

# plt.title('PSNR vs. Number of Epochs', fontsize=16)
# plt.legend(loc='lower right', fontsize=10)
# plt.grid(True)
# plt.savefig('data/denoising/Set14/mask/0/psnr_comb0.svg')

## figure -7 from set-14 dataset
# Paths to the four .npz files
# path1 = 'data/denoising/Set14/mask/7/SAM(sigma=0.1,lr=0.01,decay=0,beta=0)/det/-1.3/1e-09/out_sparsenet/0.1/psnr_7.npz'
# path2 = 'data/denoising/Set14/mask/7/ADAM(sigma=0.1,lr=0.01,decay=0,beta=0,reg=0.05)/deepdecoder/0.1/psnr_7.npz'
# path3 = 'data/denoising/Set14/mask/7/sgld/psnr_7.npz'
# path4 = 'data/denoising/Set14/mask/7/ADAM(sigma=0.1,lr=0.001,decay=0,beta=0,reg=0.05)/psnr_7.npz'
# path5 = 'data/denoising/Set14/mask/4/pat/14_0.2/trans_7_sparsenet/0.1/psnr_4.npz'

# # Loading data from the npz files
# data1 = np.load(path1)
# data2 = np.load(path2)
# data3 = np.load(path3)
# data4 = np.load(path4)
# data5 = np.load(path5)

# # Extracting the 'psnr' variable from each file
# psnr1 = data1['psnr']
# psnr2 = data2['psnr']
# psnr3 = data3['psnr']
# psnr4 = data4['psnr']
# psnr5 = data5['psnr']

# max_length = max(len(psnr1), len(psnr2), len(psnr3), len(psnr4), len(psnr5))
# epochs = np.arange(max_length) * 200 

# # Function to extend an array if needed
# def extend_array(arr, length):
#     if len(arr) < length:
#         last_val = arr[-1]
#         extended_arr = np.append(arr, [last_val] * (length - len(arr)))
#         return extended_arr
#     else:
#         return arr

# # Extending the PSNR arrays
# psnr1_extended = extend_array(psnr1, max_length)
# psnr2_extended = extend_array(psnr2, max_length)
# psnr3_extended = extend_array(psnr3, max_length)
# psnr4_extended = extend_array(psnr4, max_length)    
# psnr5_extended = extend_array(psnr5, max_length)

# # Plotting the data
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, psnr1_extended, label='Sparse-DIP')
# plt.plot(epochs, psnr2_extended, label='Deep Decoder')
# plt.plot(epochs, psnr3_extended, label='SGLD')
# plt.plot(epochs, psnr4_extended, label='Dense DIP')
# plt.plot(epochs, psnr5_extended, label='IMP')

# # Enlarging the x-axis, y-axis, and their ticks
# plt.xlabel('Number of Epochs', fontsize=14)
# plt.ylabel('PSNR', fontsize=14)
# plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.ylim(18, 25)

# plt.title('PSNR vs. Number of Epochs', fontsize=16)
# plt.legend(loc='lower right', fontsize=10)
# plt.grid(True)
# plt.savefig('data/denoising/Set14/mask/7/psnr_comb7.svg')

## figure-0 from face dataset


# def interpolate_array(arr, new_length):
#     old_indices = np.linspace(0, 1, len(arr))
#     new_indices = np.linspace(0, 1, new_length)
#     interpolated_arr = np.interp(new_indices, old_indices, arr)
#     return interpolated_arr


# path1 = 'data/denoising/face/mask/0/unet/det/selected/0.02/1e-09/out_sparsenet/0.1/psnr_0.npz'
# path2 = 'data/denoising/face/decoder/psnr_0.npz'
# path3 = 'data/denoising/face/baselines/0/sgld/psnr_0.npz'
# path4 = 'data/denoising/face/baselines/0/vanilla/psnr_0.npz'
# path5 = 'data/denoising/face/mask/0/pat/14_0.2/psnr_data_iter_13.npz'


# # Loading data from the npz files
# data1 = np.load(path1)
# data2 = np.load(path2)
# data3 = np.load(path3)
# data4 = np.load(path4)
# data5 = np.load(path5)

# # Extracting the 'psnr' variable from each file
# psnr1 = data1['psnr']
# psnr2 = data2['psnr']
# psnr3 = data3['psnr']
# psnr4 = data4['psnr']
# psnr5 = data5['psnr']


# max_length = max(len(psnr1), len(psnr2), len(psnr3), len(psnr4), len(psnr5))

# psnr2_interpolated = interpolate_array(psnr2, max_length)
# psnr5_interpolated = interpolate_array(psnr5, max_length)

# epochs = np.arange(max_length) * 200 
# print(len(psnr2),len(psnr5),len(psnr1))
# # Function to extend an array if needed
# def extend_array(arr, length):
#     if len(arr) < length:
#         last_val = arr[-1]
#         extended_arr = np.append(arr, [last_val] * (length - len(arr)))
#         return extended_arr
#     else:
#         return arr

# # Extending the PSNR arrays
# psnr1_extended = extend_array(psnr1, max_length)
# psnr2_extended = extend_array(psnr2, max_length)
# psnr3_extended = extend_array(psnr3, max_length)
# psnr4_extended = extend_array(psnr4, max_length)    
# psnr5_extended = extend_array(psnr5_interpolated, max_length)

# # Plotting the data
# plt.figure(figsize=(10, 6))
# plt.plot(epochs, psnr1_extended, label='Sparse-DIP')
# plt.plot(epochs, psnr2_extended, label='Deep Decoder')
# plt.plot(epochs, psnr3_extended, label='SGLD')
# plt.plot(epochs, psnr4_extended, label='Dense DIP')
# plt.plot(epochs, psnr5_extended, label='IMP')

# # Enlarging the x-axis, y-axis, and their ticks
# plt.xlabel('Number of Epochs', fontsize=14)
# plt.ylabel('PSNR', fontsize=14)
# plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# plt.xticks(fontsize=12)

# plt.yticks(fontsize=12)
# plt.ylim(18, 33)

# plt.title('PSNR vs. Number of Epochs', fontsize=16)
# plt.legend(loc='lower right', fontsize=10)
# plt.grid(True)
# plt.savefig('data/denoising/face/psnr_0_comb.svg')



## figure-1 from Dataset dataset


def interpolate_array(arr, new_length):
    old_indices = np.linspace(0, 1, len(arr))
    new_indices = np.linspace(0, 1, new_length)
    interpolated_arr = np.interp(new_indices, old_indices, arr)
    return interpolated_arr


path1 = 'data/denoising/Dataset/mask/1/sparsity/unet/det/0.02/1e-09/out_sparsenet/0.1/psnr_1.npz'
path2 = 'data/denoising/Dataset/mask/1/ADAM(sigma=0.1,lr=0.01,decay=0,beta=0,reg=0.05)/deepdecoder/0.1/psnr_1.npz'
path3 = 'data/denoising/Dataset/mask/1/sgld/psnr_1.npz'
path4 = 'data/denoising/Dataset/mask/1/ADAM(sigma=0.1,lr=0.001,decay=0,beta=0,reg=0.05)/psnr_1.npz'
path5 = 'data/denoising/Dataset/mask/1/pat/14_0.2/psnr_data_iter_14.npz'


# Loading data from the npz files
data1 = np.load(path1)
data2 = np.load(path2)
data3 = np.load(path3)
data4 = np.load(path4)
data5 = np.load(path5)

# Extracting the 'psnr' variable from each file
psnr1 = data1['psnr']
psnr2 = data2['psnr']
psnr3 = data3['psnr']
psnr4 = data4['psnr']
psnr5 = data5['psnr']


max_length = max(len(psnr1), len(psnr2), len(psnr3), len(psnr4), len(psnr5))

psnr2_interpolated = interpolate_array(psnr2, max_length)
psnr5_interpolated = interpolate_array(psnr5, max_length)

epochs = np.arange(max_length) * 200 
print(len(psnr2),len(psnr5),len(psnr1))
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
psnr4_extended = extend_array(psnr4, max_length)    
psnr5_extended = extend_array(psnr5_interpolated, max_length)

# Plotting the data
plt.figure(figsize=(10, 6))
plt.plot(epochs, psnr1_extended, label='Sparse-DIP')
plt.plot(epochs, psnr2_extended, label='Deep Decoder')
plt.plot(epochs, psnr3_extended, label='SGLD')
plt.plot(epochs, psnr4_extended, label='Dense DIP')
plt.plot(epochs, psnr5_extended, label='IMP')

# Enlarging the x-axis, y-axis, and their ticks
plt.xlabel('Number of Epochs', fontsize=14)
plt.ylabel('PSNR', fontsize=14)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(fontsize=12)

plt.yticks(fontsize=12)
plt.ylim(12, 30)

plt.title('PSNR vs. Number of Epochs', fontsize=16)
plt.legend(loc='lower right', fontsize=10)
plt.grid(True)
plt.savefig('data/denoising/Dataset/psnr_1_comb.svg')