from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    img_path = Image.open(path)
    img = np.array(img_path)
    return img


def edge_detection(image):
     # Convert to grayscale
    grayscale_image = np.mean(image, axis=2)

    # Define the vertical and horizontal edge detection filters
    kernelY = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    kernelX = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    
    # Apply convolution for horizontal and vertical edges
    edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)
    
    # Compute the magnitude of the edges
    edgeMAG = np.sqrt(edgeX*2 + edgeY*2)
    
    return edgeMAG
