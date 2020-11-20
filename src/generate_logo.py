# -*- coding: utf-8 -*-
# Copyright (C) by Greg Buzzard <buzzard@purdue.edu>
# All rights reserved. 

"""
Overview:
    Generate the svmbir logo.  See https://svmbir.readthedocs.io/en/latest/
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from skimage.transform import radon, rotate, iradon, rescale
from skimage import exposure

""" 
Script to load an image file with 'SVMBIR' text, apply the radon transform and gamma correction
to increase contrast, and then save the result.  
"""


def read_grey(filename, channel=0):
    """
    Read an image file and return the red component (or component specified by channel)
    of the image.  This is used to get an nxm greyscale image from a file saved in rgb or rgba format.

    Args:
        filename: Name of image file to read
        channel: Index of rgba channel to save if provided (defaults to red).

    Returns:
        The specified component of an rgb or rgba image.
    """
    img = imread(filename)
    if len(img.shape) > 2:
        img = np.squeeze(img[:, :, channel])
    return img.astype(np.float)/255


def copy_in(target, source, h_offset=0, w_offset=0, method="copy"):
    """
    Copy source array into the specified location in target array

    Args:
        target: Image to copy into
        source: Image to copy from
        h_offset: Starting x index of target
        w_offset: Starting y index of target
        method: String - either "copy" to copy the source or "max" to take the max of source and target

    Returns:
        The updated target

    """
    h = source.shape[0]
    w = source.shape[1]
    if method == "copy":
        target[h_offset:h_offset + h, w_offset:w_offset + w] = source
    else:
        target_area = target[h_offset:h_offset + h, w_offset:w_offset + w]
        target[h_offset:h_offset + h, w_offset:w_offset + w] = np.fmax(target_area, source)
    return target


def generate_logo(filename):
    """
    Load component images, apply a sinogram, and assemble to form the svmbir logo.

    Args:
        filename: Name of image file used to generate the sinogram for inclusion in the logo.

    Returns:
        None
    """
    # Load the svmbir image, convert to negative, display, and save
    image = read_grey(filename)
    image = 1-image

    plt.imshow(image, cmap=plt.cm.Greys_r)
    plt.title("Original")
    plt.show()
    svmbir_letters = np.copy(image)

    # Apply the radon transform and do gamma correction to improve contrast
    theta = np.linspace(0., 360., max(image.shape), endpoint=False)
    sinogram = radon(image, theta=theta, circle=True)
    sinogram_scaled = sinogram / np.amax(sinogram)
    gamma_corrected = exposure.adjust_gamma(sinogram_scaled, 0.4)

    # Display the sinogram and gamma corrected version
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    ax1.set_title("Radon transform\n(Sinogram)")
    ax1.set_xlabel("Projection angle (deg)")
    ax1.set_ylabel("Projection position (pixels)")
    ax1.imshow(sinogram, cmap=plt.cm.Greys_r,
               extent=(0, 360, 0, sinogram.shape[0]), aspect='auto')

    ax2.set_title("Gamma corrected")
    ax2.imshow(gamma_corrected, cmap=plt.cm.Greys_r,
               extent=(0, 360, 0, sinogram.shape[0]), aspect='auto')

    fig.tight_layout()
    plt.show()

    # Save the gamma-corrected, rotated sinogram
    sinogram_int = np.round(gamma_corrected*255)
    sinogram_copy = np.copy(sinogram_int)
    sinogram_rot = rotate(sinogram_copy, 90)
    imsave('sinogram_rot.png', sinogram_rot.astype(np.uint8))

    # Do the reconstruction and display
    # sinogram_int = imread('sinogram_rot.png')
    # sinogram_int = rotate(sinogram_int, -90)
    sinogram = exposure.adjust_gamma(sinogram_copy / 255, 2)
    image_recov = iradon(sinogram, theta=theta)

    image_recov_orig = iradon(np.round(sinogram_scaled * 255) / 255, theta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5))

    ax1.set_title("Recon from original")
    ax1.imshow(image_recov_orig, cmap=plt.cm.Greys_r, aspect='equal')

    ax2.set_title("Recon from gamma corrected")
    ax2.imshow(image_recov, cmap=plt.cm.Greys_r,aspect='equal')

    fig.tight_layout()
    plt.show()

    # Load the images for the logo
    sino = sinogram_rot.astype(float)/255 # read_grey('sinogram_rot.png')
    mbir_letters = read_grey('images/mbir.png')
    arrow_left = read_grey('images/arrow_left.png', channel=3)
    arrow_right = read_grey('images/arrow_right.png', channel=3)
    svmbir_text = read_grey('images/text.png', channel=3)

    # Set dimensions for logo elements - rescale images as needed
    spacer = 5
    pad = 10
    new_width = sino.shape[1] + svmbir_letters.shape[0]
    mbir_letters = rescale(mbir_letters, new_width / mbir_letters.shape[1])
    mbir_letters = exposure.adjust_gamma(mbir_letters, 0.2)
    mbir_letters[mbir_letters > 0.95] = 1
    mbir_letters[mbir_letters < 0.05] = 0

    new_height = sino.shape[0] + mbir_letters.shape[0] + spacer
    svmbir_text = 1 - rescale(svmbir_text, new_height / svmbir_text.shape[0])
    svmbir_text = exposure.adjust_gamma(svmbir_text, 1.5)
    svmbir_text[svmbir_text > 0.95] = 1
    svmbir_text[svmbir_text < 0.05] = 0

    height = sino.shape[0] + mbir_letters.shape[0] + 3 * spacer
    width = mbir_letters.shape[1] + svmbir_text.shape[1] + 2 * spacer

    # Get the empty image and add the sinogram
    logo = np.zeros((height, width))
    i = spacer
    j = spacer
    logo = copy_in(logo, sino, i, j)

    # Add the vertical bar and the svmbir letters with arrows
    i = spacer
    j = sino.shape[1]
    white_bar_ver = np.ones((sino.shape[1], spacer))
    logo = copy_in(logo, white_bar_ver, i, j)
    j = sino.shape[1] + spacer
    arrow_left = rescale(arrow_left, (svmbir_letters.shape[1]/2) / arrow_left.shape[1])
    logo = copy_in(logo, arrow_left, i, j)
    j = j + arrow_left.shape[1]
    arrow_right = rescale(arrow_right, (svmbir_letters.shape[1]/2) / arrow_right.shape[1])
    logo = copy_in(logo, arrow_right, i, j)
    j = sino.shape[1] + spacer
    logo = copy_in(logo, svmbir_letters, i, j, method="max")

    # Add the svmbir text
    i = spacer
    j = sino.shape[1] + svmbir_letters.shape[1] + spacer
    logo = copy_in(logo, svmbir_text, i, j)

    # Add the horizontal bar and the MBIR text
    i = spacer + sino.shape[0]
    j = spacer
    white_bar_hor = np.ones((spacer, sino.shape[1] + spacer + svmbir_letters.shape[1]))
    logo = copy_in(logo, white_bar_hor, i, j)
    i = i + spacer
    logo = copy_in(logo, mbir_letters, i, j)

    # Display the logo
    plt.imshow(logo, cmap=plt.cm.Greys_r)
    plt.show()
    imsave('logo.png', logo)


if __name__ == '__main__':

    generate_logo('images/svmbir.png')
