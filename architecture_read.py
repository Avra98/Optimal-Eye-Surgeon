## plots individual files 

# import re
# import matplotlib.pyplot as plt
# import os

# def parse_output_and_plot_histogram(file_path, figures_folder='figures_l1'):
#     # Lookup table for layer naming
#     lookup_table = {
#         '1.1.1.weight': 'downsampling-1',
#         '1.4.1.weight': 'convolution-1',
#         '1.7.1.1.1.weight': 'downsampling-2',
#         '1.7.1.4.1.weight': 'convolution-2',
#         '1.7.1.7.1.1.1.weight': 'downsampling-3',
#         '1.7.1.7.1.4.1.weight': 'convolution-3',
#         '1.7.1.7.1.7.1.1.1.weight': 'downsampling-4',
#         '1.7.1.7.1.7.1.4.1.weight': 'convolution-4',
#         '1.7.1.7.1.7.1.7.1.1.1.weight': 'downsampling-5',
#         '1.7.1.7.1.7.1.7.1.4.1.weight': 'convolution-5',
#         '1.7.1.7.1.7.1.7.1.7.1.1.1.weight': 'downsampling-6',
#         '1.7.1.7.1.7.1.7.1.7.1.4.1.weight': 'convolution-6',

#         # Continue the pattern for upsampling layers...
#         '1.7.1.7.1.7.1.7.1.7.3.1.weight': 'upsampling-6',
#         '1.7.1.7.1.7.1.7.3.1.weight': 'upsampling-5',
#         '1.7.1.7.1.7.3.1.weight': 'upsampling-4',
#         '1.7.1.7.3.1.weight': 'upsampling-3',
#         '1.7.3.1.weight': 'upsampling-2',
#         '3.1.weight': 'upsampling-1',
#         '6.1.weight': 'final convolution'
#     }

#     # Regular expression to extract weight layer names and corresponding sparsity levels
#     weight_layer_pattern = re.compile(
#         r"((?:\d+\.)+\d+\.weight)\s+\|\s+nonzeros\s+=\s+\d+\s+/\s+\d+\s+\(\s+(\d+\.\d+)%\)"
#     )

#     # Read the file content
#     try:
#         with open(file_path, 'r') as file:
#             output_data = file.read()
#     except FileNotFoundError:
#         print(f"File not found: {file_path}")
#         return

#     weight_layers = weight_layer_pattern.findall(output_data)

#     # Separating layer names and sparsity levels, translating names using the lookup table
#     layer_names = []
#     sparsity_levels = []

#     for layer, sparsity in weight_layers:
#         if layer in lookup_table:
#             layer_names.append(lookup_table[layer])
#             sparsity_levels.append(float(sparsity))

#     # Plotting the histogram
#     plt.figure(figsize=(14, 7))
#     print(sparsity_levels)
#     plt.bar(layer_names, sparsity_levels, color='blue', width=0.4)
#     plt.xlabel('Layer Type')
#     plt.ylabel('Sparsity Level log-scale(%)')
#     plt.title('Histogram of Weight Layer Sparsity')
#     plt.xticks(rotation=60)
#     plt.yscale('log')
#     plt.tight_layout()
#     figures_folder = os.path.join(os.path.dirname(file_path), 'images') 
#     # Ensure the figures_folder exists
#     if not os.path.exists(figures_folder):
#         os.makedirs(figures_folder)

#     # Define the path for saving the figure
#     figure_path = os.path.join(figures_folder, os.path.basename(file_path).replace('.out', '_sparsity_histogram.png'))

#     # Saving the figure with the name of the output file
#     plt.savefig(figure_path, bbox_inches='tight')
#     plt.close()
#     print(f"Figure saved at: {figure_path}")

# # Path to the .out file
# file_name = 'Set14_logfiles/rebtual_l1/sparsity0.05_ino8_kl1e-9_l1_p.out'  # Change to your actual .out file path

# # Call the function with the path to the .out file
# parse_output_and_plot_histogram(file_name)



### plots separately for all the files in the folder

# import re
# import matplotlib.pyplot as plt
# import os

# def parse_and_plot_histograms_in_folder(folder_path, figures_subfolder='images'):
#     # Lookup table for layer naming
#     lookup_table = {
#         '1.1.1.weight': 'downsampling-1',
#         '1.4.1.weight': 'convolution-1',
#         '1.7.1.1.1.weight': 'downsampling-2',
#         '1.7.1.4.1.weight': 'convolution-2',
#         '1.7.1.7.1.1.1.weight': 'downsampling-3',
#         '1.7.1.7.1.4.1.weight': 'convolution-3',
#         '1.7.1.7.1.7.1.1.1.weight': 'downsampling-4',
#         '1.7.1.7.1.7.1.4.1.weight': 'convolution-4',
#         '1.7.1.7.1.7.1.7.1.1.1.weight': 'downsampling-5',
#         '1.7.1.7.1.7.1.7.1.4.1.weight': 'convolution-5',
#         '1.7.1.7.1.7.1.7.1.7.1.1.1.weight': 'downsampling-6',
#         '1.7.1.7.1.7.1.7.1.7.1.4.1.weight': 'convolution-6',

#         # Continue the pattern for upsampling layers...
#         '1.7.1.7.1.7.1.7.1.7.3.1.weight': 'upsampling-6',
#         '1.7.1.7.1.7.1.7.3.1.weight': 'upsampling-5',
#         '1.7.1.7.1.7.3.1.weight': 'upsampling-4',
#         '1.7.1.7.3.1.weight': 'upsampling-3',
#         '1.7.3.1.weight': 'upsampling-2',
#         '3.1.weight': 'upsampling-1',
#         '6.1.weight': 'final convolution'
#     }

#     # Regular expression to extract weight layer names and corresponding sparsity levels
#     weight_layer_pattern = re.compile(
#         r"((?:\d+\.)+\d+\.weight)\s+\|\s+nonzeros\s+=\s+\d+\s+/\s+\d+\s+\(\s+(\d+\.\d+)%\)"
#     )

#     # Fixed y-axis scale for log plot
#     y_ticks = [10**i for i in range(-2, 3)]  # From 10^-2 to 10^2

#     # Iterate over all '.out' files in the folder_path
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith('.out'):
#             file_path = os.path.join(folder_path, file_name)

#             # Read the file content
#             try:
#                 with open(file_path, 'r') as file:
#                     output_data = file.read()
#             except FileNotFoundError:
#                 print(f"File not found: {file_path}")
#                 continue

#             weight_layers = weight_layer_pattern.findall(output_data)

#             # Separating layer names and sparsity levels, translating names using the lookup table
#             layer_names = []
#             sparsity_levels = []

#             for layer, sparsity in weight_layers:
#                 if layer in lookup_table:
#                     layer_names.append(lookup_table[layer])
#                     sparsity_levels.append(float(sparsity))

#             # Plotting the histogram
#             plt.figure(figsize=(14, 7))
#             plt.bar(layer_names, sparsity_levels, color='blue', width=0.4)
#             plt.xlabel('Layer Type')
#             plt.ylabel('Sparsity Level (%) (log-scale)')
#             plt.title(f'Histogram of Weight Layer Sparsity')
#             plt.xticks(rotation=60)
#             plt.yscale('log')
#             plt.yticks(y_ticks, [f"{tick}" for tick in y_ticks])  # Set the y-axis ticks
#             plt.tight_layout()

#             # Ensure the figures_folder exists
#             figures_folder = os.path.join(folder_path, figures_subfolder)
#             if not os.path.exists(figures_folder):
#                 os.makedirs(figures_folder)

#             # Define the path for saving the figure
#             figure_path = os.path.join(figures_folder, file_name.replace('.out', '_sparsity_histogram.png'))

#             # Saving the figure with the name of the output file
#             plt.savefig(figure_path, bbox_inches='tight')
#             plt.close()
#             print(f"Figure saved at: {figure_path}")

# # Specify the path to the folder containing the '.out' files
# folder_path = 'Set14_logfiles/rebtual_l1'  # Change to your actual folder path

# # Process all '.out' files in the folder
# parse_and_plot_histograms_in_folder(folder_path)



## process files and make conmbined histofgram 
import re
import matplotlib.pyplot as plt
import os
import numpy as np

def layer_key_to_tuple(key):
    return tuple(int(part) if part.isdigit() else part
                 for part in re.split(r'(\d+)', key))


def plot_combined_histogram(folder_path, file_names, output_folder='Set14_logfiles/rebtual_l1'):
    lookup_table = {
        '1.1.1.weight': 'downsampling-1',
        '1.4.1.weight': 'convolution-1',
        '1.7.1.1.1.weight': 'downsampling-2',
        '1.7.1.4.1.weight': 'convolution-2',
        '1.7.1.7.1.1.1.weight': 'downsampling-3',
        '1.7.1.7.1.4.1.weight': 'convolution-3',
        '1.7.1.7.1.7.1.1.1.weight': 'downsampling-4',
        '1.7.1.7.1.7.1.4.1.weight': 'convolution-4',
        '1.7.1.7.1.7.1.7.1.1.1.weight': 'downsampling-5',
        '1.7.1.7.1.7.1.7.1.4.1.weight': 'convolution-5',
        '1.7.1.7.1.7.1.7.1.7.1.1.1.weight': 'downsampling-6',
        '1.7.1.7.1.7.1.7.1.7.1.4.1.weight': 'convolution-6',

        # Continue the pattern for upsampling layers...
        '1.7.1.7.1.7.1.7.1.7.3.1.weight': 'upsampling-6',
        '1.7.1.7.1.7.1.7.3.1.weight': 'upsampling-5',
        '1.7.1.7.1.7.3.1.weight': 'upsampling-4',
        '1.7.1.7.3.1.weight': 'upsampling-3',
        '1.7.3.1.weight': 'upsampling-2',
        '3.1.weight': 'upsampling-1',
        '6.1.weight': 'final convolution'
    }

    
    legend_labels = {
        'sparsity0.05_ino8_kl1e-9_kl_p.out': 'KL',
        'sparsity0.05_ino8_kl1e-9_l1_p.out': 'L1',
        'sparsity0.05_ino8_kl1e-9_l1cent_p.out': 'L1_centered'
    }
    # Sort the keys of the lookup_table based on the order given by their corresponding values
    sorted_layer_keys = sorted(lookup_table,key=layer_key_to_tuple)

    weight_layer_pattern = re.compile(
        r"((?:\d+\.)+\d+\.weight)\s+\|\s+nonzeros\s+=\s+\d+\s+/\s+\d+\s+\(\s+(\d+\.\d+)%\)"
    )

    # Dictionary to store sparsity levels for each file
    sparsity_data = {fn: {} for fn in file_names}

    for fn in file_names:
        file_path = os.path.join(folder_path, fn)
        try:
            with open(file_path, 'r') as f:
                data = f.read()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

        matches = weight_layer_pattern.findall(data)
        for match in matches:
            layer, sparsity = match
            if layer in lookup_table:
                sparsity_data[fn][lookup_table[layer]] = float(sparsity)

    # Plotting
    TITLE_SIZE = 16
    AXIS_LABEL_SIZE = 16
    XTICKS_SIZE = 14
    YTICKS_SIZE = 20
    LEGEND_SIZE = 14
    plt.figure(figsize=(15, 8))
    bar_width = 0.2

    # X-axis positions for each group of bars
    index = np.arange(len(lookup_table))

    # Plot bars for each file
    for i, (fn, layers) in enumerate(sparsity_data.items()):
        sparsity_levels = [layers.get(lookup_table[layer], 0) for layer in sorted_layer_keys]
        plt.bar(index + i * bar_width, sparsity_levels, bar_width, label=legend_labels[fn])

    plt.xlabel('Layer Type',fontsize=AXIS_LABEL_SIZE)
    plt.ylabel('Sparsity Level (%) (log-scale)',fontsize=AXIS_LABEL_SIZE)
    plt.title('Combined Histogram of Weight Layer Sparsity for Ppt3 image', fontsize=TITLE_SIZE)
    plt.xticks(index + bar_width / 2 * len(file_names), [lookup_table[layer] for layer in sorted_layer_keys], rotation=60, fontsize=XTICKS_SIZE)
    plt.yticks(fontsize=YTICKS_SIZE) 
    plt.yscale('log')
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, 'combined_sparsity_histogram_8.png')
    plt.savefig(output_file, bbox_inches='tight')
    output_file_svg = os.path.join(output_folder, 'combined_sparsity_histogram_8.svg')
    plt.savefig(output_file_svg, bbox_inches='tight')
    plt.close()
    print(f"Combined figure saved at: {output_file}")

# Specify the folder and file names
folder_path = 'Set14_logfiles/rebtual_l1'
file_names = [
    'sparsity0.05_ino8_kl1e-9_kl_p.out',
    'sparsity0.05_ino8_kl1e-9_l1_p.out',
    'sparsity0.05_ino8_kl1e-9_l1cent_p.out'
]

plot_combined_histogram(folder_path, file_names)
