from PIL import Image
import numpy as np
from scipy.ndimage import convolve

def load_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

def edge_detection(image):
    grey = np.mean(image, axis=2)
    kernelY = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])

    kernelX = np.array([[-1, 0, 1],
                        [ -2,  0,  2],
                        [ -1,  0,  1]])
    edgeY = convolve(grey, kernelY, mode="constant", cval=0.0)
    edgeX = convolve(grey, kernelX, mode="constant", cval=0.0)
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
