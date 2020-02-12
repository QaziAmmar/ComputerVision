from skimage.exposure import rescale_intensity
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io
import numpy as np
import argparse
import cv2
from custom_classes import cv_iml


def segment_colorfulness(image, mask):
    # split the image into its respective RGB components, then mask
    # each of the individual RGB channels so we can compute
    # statistics only for the masked region
    (B, G, R) = cv2.split(image.astype("float"))
    R = np.ma.masked_array(R, mask=mask)
    G = np.ma.masked_array(B, mask=mask)
    B = np.ma.masked_array(B, mask=mask)
    # compute rg = R - G
    rg = np.absolute(R - G)
    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)
    # compute the mean and standard deviation of both `rg` and `yb`,
    # then combine them
    stdRoot = np.sqrt((rg.std() ** 2) + (yb.std() ** 2))
    meanRoot = np.sqrt((rg.mean() ** 2) + (yb.mean() ** 2))
    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


# load the image in OpenCV format so we can draw on it later, then
# allocate memory for the superpixel colorfulness visualization
image_path = "/Users/qaziammar/Downloads/IMG_4452.JPG"
# image_segments = 500
orig = cv2.imread(image_path)
vis = np.zeros(orig.shape[:2], dtype="float")
# load the image and apply SLIC superpixel segmentation to it via
# scikit-image
image = io.imread(image_path)
segments = slic(img_as_float(image), compactness=5, sigma=15)

# %%
# loop over each of the unique superpixels
for v in np.unique(segments):
    # construct a mask for the segment so we can compute image
    # statistics for *only* the masked region
    mask = np.ones(image.shape[:2])
    mask[segments == v] = 0
    # compute the superpixel colorfulness, then update the
    # visualization array
    C = segment_colorfulness(orig, mask)
    vis[segments == v] = C

# %%
# scale the visualization image from an unrestricted floating point
# to unsigned 8-bit integer array so we can use it with OpenCV and
# display it to our screen
vis = rescale_intensity(vis, out_range=(0, 255)).astype("uint8")
# overlay the superpixel colorfulness visualization on the original
# image
alpha = 0.6
overlay = np.dstack([vis] * 3)
output = orig.copy()
cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

# %%
# show the output images
cv_iml.image_show(orig)
cv_iml.image_show(vis)
cv_iml.image_show(output)
# cv2.imshow("Input", orig)
# cv2.imshow("Visualization", vis)
# cv2.imshow("Output", output)

#%%
# import matplotlib.pyplot as plt
# import numpy as np
#
# from skimage.data import astronaut
# from skimage.color import rgb2gray
# from skimage.filters import sobel
# from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
# from skimage.segmentation import mark_boundaries
# from skimage.util import img_as_float
#
# image_path = "/Users/qaziammar/Downloads/IMG_4782.jpg"
# image = cv2.imread(image_path)
#
# img = img_as_float(image)
#
# segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
# segments_slic = slic(img, n_segments=250, compactness=15, sigma=1)
# segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
# gradient = sobel(rgb2gray(img))
# segments_watershed = watershed(gradient, markers=250, compactness=0.001)
#
# print(f"Felzenszwalb number of segments: {len(np.unique(segments_fz))}")
# print(f"SLIC number of segments: {len(np.unique(segments_slic))}")
# print(f"Quickshift number of segments: {len(np.unique(segments_quick))}")
#
# fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
#
# ax[0, 0].imshow(mark_boundaries(img, segments_fz))
# ax[0, 0].set_title("Felzenszwalbs's method")
# ax[0, 1].imshow(mark_boundaries(img, segments_slic))
# ax[0, 1].set_title('SLIC')
# ax[1, 0].imshow(mark_boundaries(img, segments_quick))
# ax[1, 0].set_title('Quickshift')
# ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
# ax[1, 1].set_title('Compact watershed')
#
# for a in ax.ravel():
#     a.set_axis_off()
#
# plt.tight_layout()
# plt.show()

