import matplotlib
from skimage import io
from PIL import Image

matplotlib.use('macosx')

import matplotlib.pyplot as plt
import numpy as np

import glob


def get_classes(file_location):
    file = glob.glob(file_location)
    return plt.imread(file_location)


def get_samples(file_location):
    scan = plt.imread(file_location)  # Read file in as RGBA
    return scan[:, :, :3]  # Remove alpha/depth channel


def get_scans(folder_path):
    scans = glob.glob(folder_path + "/*.tif")
    scans.sort()

    mask = plt.imread(scans[2])
    colour = plt.imread(scans[0])
    ndvi = io.imread(scans[3])

    colour = colour[:, :, :3]  # Strip the alpha channel (depth)

    colour_shape = colour[:, :, 0].shape  # Get the y,x resolution
    colourx, coloury = colour_shape[1], colour_shape[0]  # extract the x,y resolution
    ndvi_resized = Image.fromarray(ndvi).resize((colourx, coloury))  # Resize the ndvi image to fit the data.
    ndvi_resized = np.asarray(ndvi_resized)  # convert from PIL Image to ndarray

    colour_ndvi = np.dstack((colour, ndvi_resized))  # append resized ndvi to the colour array

    return colour_ndvi, mask  # X, Y
