from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = load_image('/content/2.jpg')

# Apply median filter for noise reduction
clean_image = median(image, ball(3))

# Perform edge detection on the noise-free image
edgeMAG = edge_detection(clean_image)

# Convert edgeMAG to a binary array using a threshold
threshold = 50  # You might need to adjust this
edge_binary = edgeMAG > threshold


# Convert the boolean array to uint8 (0 and 255 for image display and saving)
edge_binary_uint8 = (edge_binary * 255).astype(np.uint8)

# Display the binary image
plt.imshow(edge_binary_uint8, cmap='gray')
plt.title('Binary Edge Image')
plt.show()


# Convert the binary array to an image and save it
edge_image = Image.fromarray(edge_binary_uint8)
edge_image.save('my_edges.png')
