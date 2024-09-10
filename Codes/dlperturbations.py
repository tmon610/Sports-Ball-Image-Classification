import os
import shutil
from functools import partial
from glob import glob

import cv2
import helper
import numpy as np
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(2022)

# Define perturbations to apply to images
perturbation = {}

# Gaussian pixel noise perturbation
perturbation["gaussian_pixel_noise"] = [
    (2 * i, partial(helper.gaussian_pixel_noise, std=2 * i)) for i in range(10)
]

# Gaussian blur perturbation
perturbation["gaussian_blur"] = [
    (i, partial(helper.gaussian_blur, num_convolve=i)) for i in range(10)
]

# Contrast increase perturbation
perturbation["contrast_increase"] = [
    (i, partial(helper.scale_contrast, scale=i))
    for i in [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25]
]

# Contrast decrease perturbation
perturbation["contrast_decrease"] = [
    (i, partial(helper.scale_contrast, scale=i))
    for i in [1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10]
]

# Brightness increase perturbation
perturbation["brightness_increase"] = [
    (i, partial(helper.change_brightness, value=i)) for i in range(0, 50, 5)
]

# Brightness decrease perturbation
perturbation["brightness_decrease"] = [
    (i, partial(helper.change_brightness, value=-i)) for i in range(0, 50, 5)
]

# Occlusion perturbation
perturbation["occlusion"] = [
    (i, partial(helper.occlusion, edge_length=i)) for i in range(0, 50, 5)
]

# Salt and pepper noise perturbation
perturbation["salt_and_pepper"] = [
    (i / 100, partial(helper.salt_and_pepper, rate=i / 100)) for i in range(0, 20, 2)
]

# Apply perturbations to images
with tqdm(total=len(perturbation)) as pbar:
    
    for perturbation_type, perturbation_fn in perturbation.items():
        
        # Remove existing perturbation directories
        shutil.rmtree(os.path.join("dataset", perturbation_type), ignore_errors=True)
        
        # Apply perturbations to images in test dataset
        for value, fn in perturbation_fn:
            for label_directory in glob(os.path.join("dataset", "test", "*")):
                label = label_directory.split("/")[-1]
                folder_dir = os.path.join(
                    "dataset", perturbation_type, str(value), label
                )
                os.makedirs(folder_dir)
                for image_directory in glob(os.path.join(label_directory, "*")):
                    
                    # Load image
                    image = helper.load_image(image_directory)
                    
                    # Apply perturbation function
                    image = fn(image)
                    image_name = image_directory.split("/")[-1]
                    image_save_dir = os.path.join(folder_dir, image_name)
                    
                    # Save perturbed image
                    helper.save_image(image, image_save_dir)
        pbar.update()