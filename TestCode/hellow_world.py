import numpy as np
from PIL import Image, ImageDraw
# This function find return the circular images of input image.
def circular_image(image_path, radius):
    image = Image.open(image_path).convert("RGB")
    npImage = np.array(image)
    h, w = image.size
    # Create same size alpha layer with circle
    alpha = Image.new('L', image.size, 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([0, 0, w, h], 0, 360, fill=255)

    # Convert alpha Image to numpy array
    npAlpha = np.array(alpha)

    # Add alpha layer to RGB
    npImage = np.dstack((npImage, npAlpha))

    # Save with alpha
    return Image.fromarray(npImage)
