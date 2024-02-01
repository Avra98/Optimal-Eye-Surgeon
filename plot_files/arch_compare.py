import re
import matplotlib.pyplot as plt
import numpy as np

# def compare_and_plot_histograms(file_name1, file_name2):
#     # Lookup table as previously defined
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
#         r"((?:\d+\.)+\d+\.weight)\s+\|\s+nonzeros\s+=\s+\d+\s+/\s+\d+\s+\(\s+(\d+\.\d+)%\)\s+\|\s+total_pruned\s+=\s+\d+\s+\|\s+shape\s+=\s+torch\.Size\(\[\d+,\s+\d+,\s+\d+,\s+\d+\]\)"
#     )

#     def extract_data(file_name):
#         with open(file_name, 'r') as file:
#             output_data = file.read()
#         weight_layers = weight_layer_pattern.findall(output_data)
#         layer_names = [lookup_table.get(layer[0], layer[0]) for layer in weight_layers if layer[0] in lookup_table]
#         sparsity_levels = [float(layer[1]) for layer in weight_layers if layer[0] in lookup_table]
#         return layer_names, sparsity_levels
    
#     def get_legend_label(file_name):
#         if 'pat' in file_name:
#             return 'IMP'
#         else:
#             return 'Sparse-DIP'

#     layer_names1, sparsity_levels1 = extract_data(file_name1)
#     layer_names2, sparsity_levels2 = extract_data(file_name2)

#     # Plotting the histogram
#     plt.figure(figsize=(14, 7))

#     # Set the positions and width for the bars
#     pos = np.arange(len(layer_names1))
#     bar_width = 0.35

#     plt.bar(pos - bar_width/2, sparsity_levels1, bar_width, label=get_legend_label(file_name1))
#     plt.bar(pos + bar_width/2, sparsity_levels2, bar_width, label=get_legend_label(file_name2))

#     plt.xlabel('Layer Type',fontsize=12)
#     plt.ylabel('Sparsity Level (%)',fontsize=12)
#     plt.title('Comparative Histogram of Weight Layer Sparsity for Lena image.\n Overall sparsity is 5%')
#     plt.xticks(pos, layer_names1, rotation=60,fontsize=12)
#     plt.legend(fontsize='large')
#     plt.tight_layout()

#     # Save or show the figure
#     # Save the figure as SVG
#     plt.savefig('comparison_histogram.svg', bbox_inches='tight')


# file_name_1 = 'trans_output/transtype_ino4_ino_trans0_transtypepat.out'
# file_name_2 = 'trans_output/transtype_ino4_ino_trans0_transtypepai.out'
# compare_and_plot_histograms(file_name_1, file_name_2)

import re
import matplotlib.pyplot as plt
import numpy as np

# Define the function to plot histograms
def compare_and_plot_histograms(*file_data):
    # Define the lookup table for layer names
    # lookup_table = {
    #     '1.1.1.weight': 'downsampling-1',
    #     '1.4.1.weight': 'convolution-1',
    #     '1.7.1.1.1.weight': 'downsampling-2',
    #     '1.7.1.4.1.weight': 'convolution-2',
    #     '1.7.1.7.1.1.1.weight': 'downsampling-3',
    #     '1.7.1.7.1.4.1.weight': 'convolution-3',
    #     '1.7.1.7.1.7.1.1.1.weight': 'downsampling-4',
    #     '1.7.1.7.1.7.1.4.1.weight': 'convolution-4',
    #     '1.7.1.7.1.7.1.7.1.1.1.weight': 'downsampling-5',
    #     '1.7.1.7.1.7.1.7.1.4.1.weight': 'convolution-5',
    #     '1.7.1.7.1.7.1.7.1.7.1.1.1.weight': 'downsampling-6',
    #     '1.7.1.7.1.7.1.7.1.7.1.4.1.weight': 'convolution-6',
    #     '1.7.1.7.1.7.1.7.1.7.3.1.weight': 'upsampling-6',
    #     '1.7.1.7.1.7.1.7.3.1.weight': 'upsampling-5',
    #     '1.7.1.7.1.7.3.1.weight': 'upsampling-4',
    #     '1.7.1.7.3.1.weight': 'upsampling-3',
    #     '1.7.3.1.weight': 'upsampling-2',
    #     '3.1.weight': 'upsampling-1',
    #     '6.1.weight': 'final convolution'
    # }
    lookup_table = {
        '1.1.weight': 'upsampling-6',
        '5.1.weight': 'upsampling-5',
        '9.1.weight': 'upsampling-4',
        '13.1.weight': 'upsampling-3',
        '17.1.weight': 'upsampling-2',
        '21.1.weight': 'upsampling-1',
        '24.1.weight': 'final convolution',
    }
    # Define the pattern to extract data from file content
    weight_layer_pattern = re.compile(
        r"((?:\d+\.)+\d+\.weight)\s+\|\s+nonzeros\s+=\s+\d+\s+/\s+\d+\s+\(\s+(\d+\.\d+)%\)\s+\|\s+total_pruned\s+=\s+\d+\s+\|\s+shape\s+=\s+torch\.Size\(\[\d+,\s+\d+,\s+\d+,\s+\d+\]\)"
    )

    # Function to extract layer names and sparsity levels from file content
    def extract_data(file_name):
        with open(file_name, 'r') as file:
            output_data = file.read()
        weight_layers = weight_layer_pattern.findall(output_data)
        layer_names = [lookup_table.get(layer[0], layer[0]) for layer in weight_layers if layer[0] in lookup_table]
        sparsity_levels = [float(layer[1]) for layer in weight_layers if layer[0] in lookup_table]
        # Order the layer names according to the lookup table
        ordered_sparsity_levels = [sparsity_levels[layer_names.index(layer)] if layer in layer_names else 0 for layer in lookup_table.values()]
        return ordered_sparsity_levels

    # Extract data for each file
    all_sparsity_levels = [extract_data(file_name) for file_name, _ in file_data]

    # Plotting the histogram
    plt.figure(figsize=(14, 7))

    # Set the positions for the bars
    pos = np.arange(len(lookup_table))
    bar_width = 0.1
    offsets = np.linspace(-bar_width * 2, bar_width * 2, len(file_data))

    # Plot bars for each file's data
    for i, (offset, (_, sparsity_label)) in enumerate(zip(offsets, file_data)):
        plt.bar(pos + offset, all_sparsity_levels[i], bar_width, label=f"{sparsity_label}% sparsity")

    plt.xlabel('Layer Type', fontsize=12)
    plt.ylabel('Sparsity Level (%)', fontsize=12)
    plt.title('Comparative Histogram of Weight Layer Sparsity for Lena image')
    plt.xticks(pos, list(lookup_table.values()), rotation=60, fontsize=12)
    plt.legend(fontsize='large')
    plt.tight_layout()

    # Save the figure

    plt.savefig('decoder_histogram_4.svg', bbox_inches='tight')

# File paths along with their respective sparsity level comments
file_data = [
    #('mask_output/kl0_ino4.out', 50),    
    ('decodermask_output/kl1e-9_prior_sigma2.0_ino4_k_decoder.out', 27),
    ('decodermask_output/kl1e-9_prior_sigma0.0_ino4_k_decoder.out', 55),
    ('decodermask_output/kl1e-9_prior_sigma-2.5_ino4_k_decoder.out', 74)
]

compare_and_plot_histograms(*file_data)





