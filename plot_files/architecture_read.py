import re
import matplotlib.pyplot as plt
import os

def parse_output_and_plot_4d_histogram(output_data, file_name, figures_folder='figures_l1'):
    # Lookup table as per your instructions
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
    # lookup_table = {
    #     '1.1.weight': 'upsampling-6',
    #     '5.1.weight': 'upsampling-5',
    #     '9.1.weight': 'upsampling-4',
    #     '13.1.weight': 'upsampling-3',
    #     '17.1.weight': 'upsampling-2',
    #     '21.1.weight': 'upsampling-1',
    #     '24.1.weight': 'final convolution',
    # }
    # Regular expression to extract weight layer names and corresponding sparsity levels
    weight_layer_pattern = re.compile(
        r"((?:\d+\.)+\d+\.weight)\s+\|\s+nonzeros\s+=\s+\d+\s+/\s+\d+\s+\(\s+(\d+\.\d+)%\)\s+\|\s+total_pruned\s+=\s+\d+\s+\|\s+shape\s+=\s+torch\.Size\(\[\d+,\s+\d+,\s+\d+,\s+\d+\]\)"
    )
    weight_layers = weight_layer_pattern.findall(output_data)

    # Separating layer names and sparsity levels, translating names using the lookup table
    layer_names = [lookup_table.get(layer[0], layer[0]) for layer in weight_layers if layer[0] in lookup_table]
    sparsity_levels = [float(layer[1]) for layer in weight_layers if layer[0] in lookup_table]

    # Plotting the histogram
    plt.figure(figsize=(14, 7))
    plt.bar(layer_names, sparsity_levels, color='blue', width=0.4)
    plt.xlabel('Layer Type')
    plt.ylabel('Sparsity Level (%)')
    plt.title('Histogram of Weight Layer Sparsity')
    plt.xticks(rotation=60)
    plt.tight_layout()
    figure_path = os.path.join(figures_folder, file_name.replace('.out', '_mapped.png'))

    # Saving the figure with the name of the output file
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close()


# def plot_histograms_for_all_out_files(folder_path):
#     # Define the subfolder name
#     figures_folder = os.path.join(folder_path, 'figures')
    
#     # Create the subfolder if it doesn't exist
#     if not os.path.exists(figures_folder):
#         os.makedirs(figures_folder)

#     # List all files in the directory and process .out files
#     for file_name in os.listdir(folder_path):
#         # Check if the file is an .out file
#         if file_name.endswith('.out'):
#             file_path = os.path.join(folder_path, file_name)
#             try:
#                 # Read the file content
#                 with open(file_path, 'r') as file:
#                     output_data = file.read()
#                 # Apply the histogram plotting function and pass the figures_folder
#                 parse_output_and_plot_4d_histogram(output_data, file_name, figures_folder)
#             except FileNotFoundError:
#                 print(f"File not found: {file_name}")
#             except Exception as e:
#                 print(f"An error occurred while processing {file_name}: {e}")



#your_folder_path = 'mask_output'
# your_folder_path = 'decodermask_output'
# plot_histograms_for_all_out_files(your_folder_path)

# Replace 'file_name' with the path to your actual .out file
file_name = 'Set14_logfiles/rebtual_l1/sparsity0.05_ino2_kl1e-9_l1.out'
#file_name = 'trans_output/transtype_ino4_ino_trans0_transtypepat.out'

# Read your file content into the output_data variable
# Ensure the file path is correct and the file is accessible
try:
    with open(file_name, 'r') as file:
        output_data = file.read()
    parse_output_and_plot_4d_histogram(output_data, file_name)
except FileNotFoundError:
    print(f"File not found: {file_name}")

