import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re

def read_psnr_from_logs(file_path, pruning_condition):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = {'epoch': [], 'psnr': []}
    capture_data = False

    for line in lines:
        if pruning_condition in line:
            capture_data = True
        elif 'PSNR of output image is:' in line:
            break  # Stop capturing if we reach the summary
        elif capture_data:
            match = re.search(r'epoch:\s+(\d+)\s+.*PSNR:\s+([\d.]+)', line)
            if match:
                epoch, psnr = match.groups()
                data['epoch'].append(int(epoch))
                data['psnr'].append(float(psnr))

    return data

def plot_psnr(data, labels):
    plt.figure(figsize=(10, 6))  # Larger figure size

    for dataset, label in zip(data, labels):
        plt.plot(dataset['epoch'], dataset['psnr'], label=label, linewidth=2)

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('PSNR', fontsize=14)
    plt.title('PSNR vs Epochs for transfer masks', fontsize=16)
    plt.grid(True)  # Enable grid
    plt.legend(fontsize=12)  # Increase legend label size
    plt.xticks(fontsize=12)  # Larger x-tick labels
    plt.yticks(fontsize=12)  # Larger y-tick labels
    plt.ylim([18, 30])  # Set y-axis limits
    plt.tight_layout()  # Adjust the padding between and around subplots
    plt.savefig('Set2_Trans_data1/combined_psnr_plot.png', dpi=300)  # High resolution
    plt.show()

# set-1 paths and files 
# file_paths = [
#     'set1_Trans/pai_dataset1_tran.out',
#     'set1_Trans/pat_dataset1_tran.out',
#     'set1_Trans/pai_self.out',
#     'set1_Trans/pat_self.out',
#     'set1_Trans/pai_face0_tran.out',
#     'set1_Trans/pat_face0_tran.out'
# ]
# pruning_conditions = [
#     '97.00% pruned',
#     '94.51% pruned',
#     '98.00% pruned',
#     '94.51% pruned',
#     '97.00% pruned',
#     '94.51% pruned'
# ]
# labels = ['OES-mask (cross dataset) ', 'IMP-mask (cross-dataset)', 'OES-mask-self', 'IMP-mask-self', 'OES-mask (inter dataaset) ', 'IMP-mask (inter dataaset)']

#set-2 paths and files 
file_paths = [
    'Set2_Trans_data1/pai_trans_data3.out',
    'Set2_Trans_data1/pat_trans_data3.out',
    'Set2_Trans_data1/pai_self.out',
    'Set2_Trans_data1/pat_self.out',
    'Set2_Trans_data1/pai_trans_face0.out',
    'Set2_Trans_data1/pat_trans_face0.out'
]
pruning_conditions = [
    '93.10% pruned',
    '94.51% pruned',
    '98.00% pruned',
    '94.51% pruned',
    '98.00% pruned',
    '94.51% pruned'
]


labels = ['OES-mask (inter-dataset)', 'IMP-mask (inter-dataset)','OES-mask-self' , 'IMP-mask-self', 'OES-mask (cross dataset) ', 'IMP-mask (cross dataset)']

# set-3 paths and files  
# file_paths = [
#     'set3_Trans/pai_dataset1.out',
#     'set3_Trans/pat_dataset.out',
#     'set3_Trans/pai_self.out',
#     'set3_Trans/pat_self.out',
#     'set3_Trans/pai_face0.out',
#     'set3_Trans/pat_face0.out'
# ]
# pruning_conditions = [
#     '95.00% pruned',
#     '94.51% pruned',
#     '97.00% pruned',
#     '94.51% pruned',
#     '97.00% pruned',
#     '94.51% pruned'
# ]    

# labels = ['OES-mask-self ', 'IMP-mask (cross-dataset)', 'OES-mask (cross dataset) ', 'IMP-mask-self', 'OES-mask (inter dataaset) ', 'IMP-mask (inter dataaset)']
    
# set-4 path and files    
# file_paths = [
#     'set4_Trans/pai_dataset1.out',
#     'set4_Trans/pat_dataset1.out',
#     'set4_Trans/pai_self.out',
#     'set4_Trans/pat_self.out',
#     'set4_Trans/pai_face0.out',
#     'set4_Trans/pat_face0.out'
# ]
# pruning_conditions = [
#     '98.00% pruned',
#     '94.51% pruned',
#     '97.00% pruned',
#     '94.51% pruned',
#     '98.00% pruned',
#     '94.51% pruned'
# ]    

# labels = ['OES-mask-self ', 'IMP-mask (cross-dataset)', 'OES-mask (cross dataset) ', 'IMP-mask-self', 'OES-mask (inter dataaset) ', 'IMP-mask (inter dataaset)']


data_sets = [read_psnr_from_logs(file_path, condition) for file_path, condition in zip(file_paths, pruning_conditions)]
plot_psnr(data_sets, labels)



