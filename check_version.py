import importlib

def get_module_version(module_name):
  """
  Attempts to import the specified module and returns its version string.
  Returns None if the module is not found.
  """
  try:
    module = importlib.import_module(module_name)
    return module.__version__
  except ModuleNotFoundError:
    return None

# Check versions of numpy, scipy, matplotlib, pillow, tqdm, scikit-image, and torchvision
numpy_version = get_module_version("numpy")
scipy_version = get_module_version("scipy")
matplotlib_version = get_module_version("matplotlib")
pillow_version = get_module_version("pillow")
tqdm_version = get_module_version("tqdm")
scikit_version = get_module_version("scikit-image")
torchvision_version = get_module_version("torchvision")  # Added torchvision

# Print the versions
print(f"numpy: {numpy_version}")
print(f"scipy: {scipy_version}")
print(f"matplotlib: {matplotlib_version}")
print(f"pillow: {pillow_version}")
print(f"tqdm: {tqdm_version}")
print(f"scikit-image: {scikit_version}")
print(f"torchvision: {torchvision_version}")  # Added torchvision
