import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from .common_utils import *
import glob
import os
import numpy as np


def get_text_mask(for_image, sz=20):
    font_fname = '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf'
    font_size = sz
    font = ImageFont.truetype(font_fname, font_size)

    img_mask = Image.fromarray(np.array(for_image)*0+255)
    draw = ImageDraw.Draw(img_mask)
    draw.text((128, 128), "hello world", font=font, fill='rgb(0, 0, 0)')

    return img_mask

# def get_bernoulli_mask(for_image, zero_fraction=0.95):
#     img_mask_np=(np.random.random_sample(size=pil_to_np(for_image).shape) > zero_fraction).astype(int)
#     img_mask = np_to_pil(img_mask_np)
    
#     return  img_mask, img_mask_np

def get_bernoulli_mask(for_image, zero_fraction=0.95):
    # Assuming for_image is a PIL image, convert it to numpy and get its shape
    img_np = pil_to_np(for_image)
    channels, height, width = img_np.shape

    # Generate a single-channel mask
    single_channel_mask = (np.random.rand(height, width) > zero_fraction).astype(int)

    # Stack the single-channel mask to create a mask with the same number of channels as the image
    img_mask_np = np.array([single_channel_mask]*channels)

    # Convert numpy array to PIL image
    img_mask = np_to_pil(img_mask_np)

    return img_mask, img_mask_np





def resize_and_crop(img_pil, base_size):
    """ Resize and crop the image to make it square with the specified base size. """
    width, height = img_pil.size

    # Find the nearest base size
    if abs(base_size - 256) < abs(base_size - 480):
        nearest_base = 256 if abs(base_size - 256) < abs(base_size - 512) else 512
    else:
        nearest_base = 480 if abs(base_size - 480) < abs(base_size - 512) else 512

    # Resize the image to maintain aspect ratio
    ratio = min(nearest_base / width, nearest_base / height)
    new_width = round(width * ratio)
    new_height = round(height * ratio)

    # Adjust dimensions if they are off due to rounding
    if new_width not in [256, 480, 512]:
        new_width = nearest_base
    if new_height not in [256, 480, 512]:
        new_height = nearest_base

    #img_pil = img_pil.resize((new_width, new_height), Image.ANTIALIAS)
    img_pil = img_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
    

    # Crop to make it square
    crop_width = (new_width - nearest_base) // 2
    crop_height = (new_height - nearest_base) // 2
    img_pil = img_pil.crop((crop_width, crop_height, new_width - crop_width, new_height - crop_height))

    return img_pil

