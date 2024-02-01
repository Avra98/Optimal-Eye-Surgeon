from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_and_process_image(image_path):
    # Load the image
    image = Image.open(image_path)
    # Convert the image to RGB if it's not
    rgb_image = image.convert('RGB')
    # Normalize the image
    normalized_image = np.array(rgb_image) / 255.0   
    return normalized_image

def compute_fourier_transform(channel_data):
    fourier_transform = np.fft.fft2(channel_data)
    fourier_transform_shifted = np.fft.fftshift(fourier_transform)
    magnitude_spectrum = np.abs(np.log(np.abs(fourier_transform_shifted) + 1e-6))
    return magnitude_spectrum

# Placeholder for the actual image path
image_path = 'average_power_spectrum_sparse.png'

# Load and process the image
normalized_image = load_and_process_image(image_path)

# Initialize an empty array for the combined Fourier transform
combined_magnitude_spectrum = np.zeros_like(normalized_image[:,:,0])

# Compute the Fourier transform for each channel
for i in range(3): # Loop over RGB channels
    magnitude_spectrum = compute_fourier_transform(normalized_image[:,:,i])
    combined_magnitude_spectrum += magnitude_spectrum

# Average the combined spectrum
combined_magnitude_spectrum /= 3

# Normalize the magnitude spectrum
normalized_spectrum = combined_magnitude_spectrum / np.max(combined_magnitude_spectrum)

# Plotting the Fourier Transform
plt.figure(figsize=(12, 6))
plt.imshow(normalized_spectrum, cmap='gray')
plt.title('Fourier Transform of RGB Image')
plt.show()

# Save the Fourier transform image
plt.imsave('fourier_transform_sparse.png', normalized_spectrum)
