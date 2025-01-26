from image_utils.py import load_image, edge_detection
from skimage.filters import median
from skimage.morphology import ball

dogs = load_image('dogs.jpg')
clean_image = median(dogs, ball(3))
