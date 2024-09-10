import numpy as np
from PIL import Image

# Read an image from the specified path using the Python Imaging Library. If mono = True then read the image as a monochrome (grayscale) image otherwise reads the image as a regular RGB or RGBA image. Finally, it converts the image data into a NumPy array and return the array representation of the image.

def read_img(path, mono=False):
    if mono:
        return read_img_mono(path)
    img = Image.open(path)
    return np.asarray(img)

# Read an image from the specified path and converts it to a monochrome (grayscale) image. uses the PIL to open the image and then converts it to monochrome using the .convert() method with the "L" mode, which stands for luminance (grayscale). Finally, it converts the monochrome image data into a NumPy array

def read_img_mono(path):
    # The L flag converts it to 1 channel.
    img = Image.open(path).convert(mode="L")
    return np.asarray(img)

# resizes an image represented by a NumPy array to the specified size. creates a PIL Image object from the input NumPy array, ensuring that pixel values are clipped to the valid range of 0 to 255 and cast to uint8 datatype. Then, it resizes the image specifying the target size as a 2-tuple (width, height). Finally, it converts the resized image back to a NumPy array. 

def resize_img(ndarray, size):
    # Parameter "size" is a 2-tuple (width, height).
    img = Image.fromarray(ndarray.clip(0, 255).astype(np.uint8))
    return np.asarray(img.resize(size))

# converts an RGB image represented by a NumPy array to a monochrome (grayscale) image. creates a PIL Image object from the input RGB NumPy array. Then, it converts the image to monochrome using the .convert() method with the "L" mode. Finally, it converts the monochrome image data into a NumPy array and returns the array representation of the grayscale image.
    
def rgb_to_gray(ndarray):
    gray_img = Image.fromarray(ndarray).convert(mode="L")
    return np.asarray(gray_img)

# displays an image represented by a NumPy array.
    
def display_img(ndarray):
    Image.fromarray(ndarray.clip(0, 255).astype(np.uint8)).show()

# saves an image represented by a NumPy array to the specified path

def save_img(ndarray, path):
    Image.fromarray(ndarray.clip(0, 255).astype(np.uint8)).save(path)
