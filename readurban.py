import os
import glob
import argparse
from PIL import Image
import random
import shutil

# Function to delete files with a specific string segment in their name
def delete_files(folder_path, string_segment):
    for file_path in glob.glob(os.path.join(folder_path, '*.png')):
       
        if string_segment in os.path.basename(file_path):
            os.remove(file_path)

# Function to print the dimensions of PNG images
def print_dimensions(folder_path):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print("The folder path does not exist.")
        return

    # Get the list of PNG files
    png_files = glob.glob(os.path.join(folder_path, '*.png'))
    
    # Check if there are any PNG files
    if not png_files:
        print("No PNG files found in the folder.")
        return

    for file_path in png_files:
        print("In this function")
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                print(f"Dimensions of {os.path.basename(file_path)}: {width}x{height}")
        except Exception as e:
            print(f"Couldn't open {file_path}. Error: {e}")

def count_images(folder_path):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print("The folder path does not exist.")
        return

    # Get the list of PNG files in the directory
    image_files = glob.glob(os.path.join(folder_path, '*.png'))
    
    # Check if there are any PNG files
    if not image_files:
        print("No PNG files found in the folder.")
        return

    # Return the count of image files
    return len(image_files)



# Function to crop and convert images to grayscale
def crop_and_grayscale(folder_path):
    for file_path in glob.glob(os.path.join(folder_path, '*.png')):
        with Image.open(file_path) as img:
            width, height = img.size
            
            # Calculate the area to crop based on the center of the image
            left = (width - 256)/2
            top = (height - 256)/2
            right = (width + 256)/2
            bottom = (height + 256)/2
            
            # Crop the image
            img = img.crop((left, top, right, bottom))
            
            # Convert the image to grayscale
            img = img.convert("L")
            
            # Save the new image
            img.save(file_path)

def split_dataset(folder_path, ratio=0.8):
    # Get the list of PNG files in the directory
    image_files = glob.glob(os.path.join(folder_path, '*.png'))
    
    # Randomly shuffle the list of image files
    random.shuffle(image_files)
    
    # Calculate the number of images for training
    train_count = int(ratio * len(image_files))

    # Split the list of files
    train_files = image_files[:train_count]
    test_files = image_files[train_count:]
    
    # Create train and test folders
    train_folder = os.path.join(folder_path, 'train')
    test_folder = os.path.join(folder_path, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    
    # Move the files
    for file in train_files:
        shutil.move(file, train_folder)
    for file in test_files:
        shutil.move(file, test_folder)            

# Command-line arguments parsing
parser = argparse.ArgumentParser()
parser.add_argument('folder_path')
parser.add_argument('--delete_files', action='store_true')
parser.add_argument('--print_dimensions', action='store_true')
parser.add_argument('--crop_and_grayscale', action='store_true')
parser.add_argument('--count_images', action='store_true')
parser.add_argument('--segment', default='')
parser.add_argument('--split_dataset', action='store_true')

args = parser.parse_args()

# Call functions based on command-line arguments
if args.delete_files:
    delete_files(args.folder_path, args.segment)
if args.print_dimensions:
    print_dimensions(args.folder_path)
if args.crop_and_grayscale:
    crop_and_grayscale(args.folder_path)
if args.count_images:
    num_images = count_images(args.folder_path)
    print(f"Number of PNG images in the directory: {num_images}")
if args.split_dataset:
    split_dataset(args.folder_path)    

