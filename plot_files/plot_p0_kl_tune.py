import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ino=0

# # Paths to the four .npz files
# path1 = f'data/denoising/Set14/mask/0/diff_p0/det/0.03/05/1e-09/out_sparsenet/0.1/psnr_0.npz'
# path2 = f'data/denoising/Set14/mask/0/diff_p0/det/0.05/05/1e-09/out_sparsenet/0.1/psnr_0.npz'
# path3 = f'data/denoising/Set14/mask/0/diff_p0/det/0.5/05/1e-09/out_sparsenet/0.1/psnr_0.npz'
# path4 = f'data/denoising/Set14/mask/0/diff_p0/det/0.8/05/1e-09/out_sparsenet/0.1/psnr_0.npz'

# # Loading data from the npz files
# data1 = np.load(path1)
# data2 = np.load(path2)
# data3 = np.load(path3)
# data4 = np.load(path4)
# # data5 = np.load(path5)
# # data6 = np.load(path6)

# # Extracting the 'psnr' variable from each file
# psnr1 = data1['psnr']
# psnr2 = data2['psnr']
# psnr3 = data3['psnr']
# psnr4 = data4['psnr']
# # psnr5 = data5['psnr']
# # psnr6 = data6['psnr']

# max_length = max(len(psnr1), len(psnr2), len(psnr3), len(psnr4))
# epochs = np.arange(max_length) * 200 
# #epochs5 = np.arange(len(psnr5)) * 1000 

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
# # psnr5_extended = extend_array(psnr5, max_length)
# # psnr6_extended = extend_array(psnr6, max_length)
# # Plotting the data
# plt.figure(figsize=(10, 6))
# # plt.plot(epochs, psnr1_extended, label=' -0.1 ')
# plt.plot(epochs, psnr1_extended, label='p0= 0.03')
# plt.plot(epochs, psnr2_extended, label='p0= 0.05')
# plt.plot(epochs, psnr3_extended, label='p0= 0.5')
# plt.plot(epochs, psnr4_extended, label='p0= 0.8')
# #plt.plot(epochs, psnr6_extended, label='Vanilla DIP')
# # Enlarging the x-axis, y-axis, and their ticks
# plt.xlabel('Number of Epochs', fontsize=14)
# plt.ylabel('PSNR', fontsize=14)
# plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# #plt.ylim(22, 31)


# plt.title('PSNR vs. Number of Epochs', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.savefig(f'data/denoising/Set14/mask/0/diff_p0/psnr_comb{ino}.svg')
# plt.savefig(f'data/denoising/Set14/mask/0/diff_p0/psnr_comb{ino}.png')

# ino=0

# # Paths to the four .npz files
# # path1 = f'data/denoising/Set14/mask/0/pat/early/14_0.2/trans_0_sparsenet_set14/0.1/psnr_0.npz'
# # path2 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/out_sparsenet/0.1/psnr_0.npz'
# # path3 = f'data/denoising/Set14/mask/0/pat/14_0.2/trans_0_sparsenet/0.1/psnr_0.npz'
# # path4 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/early/out_sparsenet/0.1/psnr_0.npz'


# path1 = f'data/denoising/Set14/mask/0/pat/early/14_0.2/trans_2_sparsenet_set14/0.1/psnr_0.npz'
# path2 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/trans_2_sparsenet_set14/0.1/psnr_0.npz'
# path3 = f'data/denoising/Set14/mask/0/pat/14_0.2/trans_2_sparsenet_set14/0.1/psnr_0.npz'
# path4 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/early/out_sparsenet/0.1/psnr_0.npz'


# # Loading data from the npz files
# data1 = np.load(path1)
# data2 = np.load(path2)
# data3 = np.load(path3)
# data4 = np.load(path4)



# # data5 = np.load(path5)
# # data6 = np.load(path6)

# # Extracting the 'psnr' variable from each file
# psnr1 = data1['psnr']
# psnr2 = data2['psnr']
# psnr3 = data3['psnr']
# psnr4 = data4['psnr']
# # psnr5 = data5['psnr']
# # psnr6 = data6['psnr']
# print(psnr1.shape)
# max_length = max(len(psnr1), len(psnr2), len(psnr3), len(psnr4))
# epochs = np.arange(max_length) * 200 
# epochs1 = np.arange(len(psnr1)) * 1000 

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
# # psnr5_extended = extend_array(psnr5, max_length)
# # psnr6_extended = extend_array(psnr6, max_length)
# # Plotting the data
# plt.figure(figsize=(10, 6))
# # plt.plot(epochs, psnr1_extended, label=' -0.1 ')
# plt.plot(epochs1, psnr1, label='IMP at early stop-time')
# plt.plot(epochs, psnr2_extended, label='OES at initialziation')
# plt.plot(epochs, psnr3_extended, label='IMP at convergence')
# #plt.plot(epochs, psnr4_extended, label='OES at early-stop time')
# #plt.plot(epochs, psnr6_extended, label='Vanilla DIP')
# # Enlarging the x-axis, y-axis, and their ticks
# plt.xlabel('Number of Epochs', fontsize=14)
# plt.ylabel('PSNR', fontsize=14)
# plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# #plt.ylim(22, 31)
# plt.xlim(0, 20000)

# plt.title('PSNR vs. Number of Epochs', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.savefig(f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/early/impoesearly_cross.png')
# plt.savefig(f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/early/impoesearly_cross.svg')

# ino=6
# path1 = f'data/denoising/Set14/mask/{ino}/sparsity/det/0.05/1e-09/early/out_sparsenet/0.1/psnr_{ino}.npz'
# path2 = f'data/denoising/Set14/mask/{ino}/sparsity/det/0.05/1e-09/out_sparsenet/0.1/psnr_{ino}.npz'

# # Loading data from the npz files
# data1 = np.load(path1)
# data2 = np.load(path2)
# # data3 = np.load(path3)
# # data4 = np.load(path4)
# # data5 = np.load(path5)
# # data6 = np.load(path6)

# # Extracting the 'psnr' variable from each file
# psnr1 = data1['psnr']
# psnr2 = data2['psnr']
# # psnr3 = data3['psnr']
# # psnr4 = data4['psnr']
# # psnr5 = data5['psnr']
# # psnr6 = data6['psnr']

# max_length = max(len(psnr1), len(psnr2))
# epochs = np.arange(max_length) * 200 
# #epochs5 = np.arange(len(psnr5)) * 1000 

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
# # psnr3_extended = extend_array(psnr3, max_length)
# # psnr4_extended = extend_array(psnr4, max_length)    
# # psnr5_extended = extend_array(psnr5, max_length)
# # psnr6_extended = extend_array(psnr6, max_length)
# # Plotting the data
# plt.figure(figsize=(10, 6))
# # plt.plot(epochs, psnr1_extended, label=' -0.1 ')
# plt.plot(epochs, psnr1_extended, label='Xavier normal initialization')
# plt.plot(epochs, psnr2_extended, label='He uniform initialization')
# # plt.plot(epochs, psnr3_extended, label='Baboon σ=25 dB')
# # plt.plot(epochs, psnr4_extended, label='Babbon σ=50 dB')
# #plt.plot(epochs, psnr6_extended, label='Vanilla DIP')
# # Enlarging the x-axis, y-axis, and their ticks
# plt.xlabel('Number of Epochs', fontsize=14)
# plt.ylabel('PSNR', fontsize=14)
# plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# #plt.ylim(22, 31)
# #plt.xlim(0, 20000)

# plt.title('PSNR vs. Number of Epochs', fontsize=16)
# plt.legend(fontsize=12)
# plt.grid(True)
# plt.savefig(f'data/denoising/Set14/mask/{ino}/sparsity/det/0.05/1e-09/early/out_sparsenet/0.1/init_compare.svg')
# plt.savefig(f'data/denoising/Set14/mask/{ino}/sparsity/det/0.05/1e-09/early/out_sparsenet/0.1/init_compare.png')


ino=0

# Paths to the four .npz files
# path1 = f'data/denoising/Set14/mask/0/pat/early/14_0.2/trans_0_sparsenet_set14/0.1/psnr_0.npz'
# path2 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/out_sparsenet/0.1/psnr_0.npz'
# path3 = f'data/denoising/Set14/mask/0/pat/14_0.2/trans_0_sparsenet/0.1/psnr_0.npz'
# path4 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/early/out_sparsenet/0.1/psnr_0.npz'


# path1 = f'data/denoising/Set14/mask/0/pat/early/14_0.2/trans_2_sparsenet_set14/0.1/psnr_0.npz'
# path2 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/trans_2_sparsenet_set14/0.1/psnr_0.npz'
# path3 = f'data/denoising/Set14/mask/0/pat/14_0.2/trans_2_sparsenet_set14/0.1/psnr_0.npz'
# path4 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/early/out_sparsenet/0.1/psnr_0.npz'


path1 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/out_sparsenet/0.1/psnr_0.npz'
path2 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/5_times/out_sparsenet/0.1/psnr_0.npz'
path3 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/01_times/out_sparsenet/0.1/psnr_0.npz'
path4 = f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/early/out_sparsenet/0.1/psnr_0.npz'



# Loading data from the npz files
data1 = np.load(path1)
data2 = np.load(path2)
data3 = np.load(path3)
data4 = np.load(path4)



# data5 = np.load(path5)
# data6 = np.load(path6)

# Extracting the 'psnr' variable from each file
psnr1 = data1['psnr']
psnr2 = data2['psnr']
psnr3 = data3['psnr']
psnr4 = data4['psnr']
# psnr5 = data5['psnr']
# psnr6 = data6['psnr']
print(psnr1.shape)
max_length = max(len(psnr1), len(psnr2), len(psnr3), len(psnr4))
epochs = np.arange(max_length) * 200 
epochs2 = np.arange(len(psnr2)) * 1000 

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
# psnr5_extended = extend_array(psnr5, max_length)
# psnr6_extended = extend_array(psnr6, max_length)
# Plotting the data
plt.figure(figsize=(10, 6))
# plt.plot(epochs, psnr1_extended, label=' -0.1 ')
plt.plot(epochs, psnr1, label='He initialization ')
plt.plot(epochs2, psnr2, label=' He initialization x5')
plt.plot(epochs, psnr3_extended, label='He initialization x0.1')
#plt.plot(epochs, psnr4_extended, label='OES at early-stop time')
#plt.plot(epochs, psnr6_extended, label='Vanilla DIP')
# Enlarging the x-axis, y-axis, and their ticks
plt.xlabel('Number of Epochs', fontsize=14)
plt.ylabel('PSNR', fontsize=14)
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.ylim(22, 31)
#plt.xlim(0, 20000)

plt.title('PSNR vs. Number of Epochs', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)
plt.savefig(f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/5_times/init_scale.png')
plt.savefig(f'data/denoising/Set14/mask/0/sparsity/det/0.05/1e-09/5_times/init_scale.svg')