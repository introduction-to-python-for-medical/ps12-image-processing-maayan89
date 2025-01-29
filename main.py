
from PIL import Image
import numpy as np
from scipy.ndimage import convolve
from image_utils.py import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball
import matplotlib.pyplot as plt

def suppress_noise(Image):
  clean_img = median(Image, ball(3))
  return clean_img

def detect_edges(Img):
  edges = edge_detection(Img)
  return edges

def binary_img(edges, threshold):
  binary_image = (edges > threshold).astype(np.uint)
  return binary_image

def save_binary_img (Img, Img_name):
  edge_img = Image.fromarray(Img*255)
  edge_img.save(Img_name)

image_array = load_image('dogs.jpg')
clean_img = suppress_noise(image_array)
edges = detect_edges(clean_img)
binary_edges = binary_img(edges, threshold=50)
save_binary_img(binary_edges, 'my_edges.png')

plt.imshow(binary_edges, cmap='gray')
plt.title("binary edges")
plt.show()








