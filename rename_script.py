import os

def rename_in_directory(directory_path, old_substring, new_substring):
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return
    
    print(f"Walking through directory: {directory_path}")
    # Walk through directory
    for dirpath, dirnames, filenames in os.walk(directory_path):
        print(f"Current directory path: {dirpath}")
        print(f"Directories: {dirnames}")
        print(f"Files: {filenames}")

        # Rename directories
        for dirname in dirnames:
            if old_substring in dirname:
                old_path = os.path.join(dirpath, dirname)
                new_dirname = dirname.replace(old_substring, new_substring)
                new_path = os.path.join(dirpath, new_dirname)
                print(f"Renaming directory: {old_path} -> {new_path}")
                os.rename(old_path, new_path)
        
        # After renaming directories, update dirnames to reflect changes
        dirnames[:] = [dirname.replace(old_substring, new_substring) if old_substring in dirname else dirname for dirname in dirnames]

        # Rename files
        for filename in filenames:
            if old_substring in filename:
                old_path = os.path.join(dirpath, filename)
                new_filename = filename.replace(old_substring, new_substring)
                new_path = os.path.join(dirpath, new_filename)
                print(f"Renaming file: {old_path} -> {new_path}")
                os.rename(old_path, new_path)

# Usage example

old_substring = '13'
new_substring = 'man'
directory_path = f'data/denoising/Set14/mask/{new_substring}/sparsity/det/0.05/1e-09'
rename_in_directory(directory_path, old_substring, new_substring)
