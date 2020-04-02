import cv2
import numpy as np
from custom_classes import cv_iml, path

# Trying to implement this paper.
# ADAPTIVE GRAY WORLD-BASED COLOR NORMALIZATION OF THIN BLOOD FILM IMAGES

folder_path = "Malaria_dataset/malaria/4a_001.jpg"
illumenation_image_path = path.dataset_path + "Malaria_dataset/4a_005.jpg"
dataset_path = path.dataset_path + folder_path

kernel = np.ones((60, 60), np.uint8)
opening_kernal = np.ones((2, 2), np.uint8)
img = cv2.imread(dataset_path)
cv_iml.image_show(img)
# %%
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_inverse = 255 - gray
# find frequency of pixels in range 0-255
ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
cv_iml.image_show(gray_inverse, 'gray')
# Calculate area ganulomatry.
# cv_iml.show_histogram(gray_inverse)
# cv_iml.image_show(thresh, 'gray')
# %%
# This code work correct for illumination correction.
closing_kernal = np.ones((190, 190), np.uint8)
illum_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, closing_kernal)
enhance_image = img - illum_img
enhance_gray = cv2.cvtColor(enhance_image, cv2.COLOR_BGR2GRAY)
cv_iml.image_show(enhance_image)
cv_iml.show_histogram(enhance_gray)

# %%
# (f) binary foreground mask
# If we apply gaussian smoothing on the image before OTSU thresholding it gives more good results.
ret, enhance_thresh = cv2.threshold(enhance_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
enhance_thresh = np.invert(enhance_thresh)
# After removing noise form image.
enhance_thresh = cv2.morphologyEx(enhance_thresh, cv2.MORPH_OPEN, opening_kernal)
# this give us the foreground pixels of image.
cv_iml.image_show(enhance_thresh, 'gray')

#%%

# After separating the input (Iui ) channels (i ∈ {r, g, b}),
b, g, r = cv2.split(img)

image_size = img.size / 3  # divide by 3 to get the number of image PIXELS

# Foreground (If ui ).
image_foreground_unknown_blue = cv2.bitwise_and(b, enhance_thresh)
image_foreground_unknown_green = cv2.bitwise_and(g, enhance_thresh)
image_foreground_unknown_red = cv2.bitwise_and(r, enhance_thresh)

# background (Ibui ) images.
background_thresh = np.invert(enhance_thresh)

image_background_unknown_blue = cv2.bitwise_and(b, background_thresh)
image_background_unknown_green = cv2.bitwise_and(g, background_thresh)
image_background_unknown_red = cv2.bitwise_and(r, background_thresh)

image_background_unknown = cv2.merge((image_background_unknown_blue, image_background_unknown_green,
                                      image_background_unknown_red))
image_foreground_unknown = cv2.merge((image_foreground_unknown_blue, image_foreground_unknown_green,
                                      image_foreground_unknown_red))
cv_iml.image_show(cv2.cvtColor(image_background_unknown, cv2.COLOR_BGR2RGB))
cv_iml.image_show(cv2.cvtColor(image_foreground_unknown, cv2.COLOR_BGR2RGB))
# %%

# Calculate Mb using (Ibui ) channel averages:
# convert to float, as B, G, and R will otherwise be int
image_background_unknown_blue_mean = image_background_unknown_blue.mean()
image_background_unknown_green_mean = image_background_unknown_green.mean()
image_background_unknown_red_mean = image_background_unknown_red.mean()

# mi= 255 / mean_Ibui
m_background_blue = 255 / image_background_unknown_blue_mean
m_background_green = 255 / image_background_unknown_green_mean
m_background_red = 255 / image_background_unknown_red_mean


# complete transformation matrix

M_background = np.array([
    [m_background_blue, 0, 0],
    [0, m_background_green, 0],
    [0, 0, m_background_red]
])

# Create an empty image and add each individual channel into.

temp_blue = m_background_blue * b
temp_green = m_background_green * g
temp_red = m_background_red * r

# 2. Transform the whole image: I1 = Mb * Iu
I1 = cv2.filter2D(img, -1, M_background)
# I1 = cv2.merge((temp_blue, temp_green, temp_red))
# cv_iml.image_show(I1)
cv_iml.image_show(I1)

# %%
# step3. Calculate Mf : mfi using Eq. 3 with (If 1i ) and the refcerence image foreground
# channels If i .
I1_blue, I1_green, I1_red = cv2.split(I1)
# Extract foreground channel of I1 image.
I1_foreground_blue = cv2.bitwise_and(I1_blue, enhance_thresh)
I1_foreground_green = cv2.bitwise_and(I1_green, enhance_thresh)
I1_foreground_red = cv2.bitwise_and(I1_red, enhance_thresh)

I1_foreground_blue_mean = I1_foreground_blue.mean()
I1_foreground_green_mean = I1_foreground_green.mean()
I1_foreground_red_mean = I1_foreground_red.mean()

I1_foreground = cv2.merge((I1_foreground_blue, I1_foreground_green,
                           I1_foreground_red))
cv_iml.image_show(I1_foreground)

# %%

# image_foreground_unknown_blue_mean = image_foreground_unknown_blue.mean()
# image_foreground_unknown_green_mean = image_foreground_unknown_green.mean()
# image_foreground_unknown_red_mean = image_foreground_unknown_red.mean()

# By equation 3 in paper
m_foreground_blue = 255 / I1_foreground_blue_mean
m_foreground_green = 255 / I1_foreground_green_mean
m_foreground_red = 255 / I1_foreground_red_mean

M_foreground = np.array([
    [m_foreground_blue, 0, 0],
    [0, m_foreground_green, 0],
    [0, 0, m_foreground_red]
])

# 4. Transform only the foreground channels: If2 = MfIf1
I2_foreground = cv2.filter2D(I1_foreground, -1, M_foreground)
cv_iml.image_show(I2_foreground)

# %%
# 5. Replace the foreground channels of I1 with If2
# it tells the locations where we want to chane the pixel values
# enhance_thresh
I2 = I1.copy()
rows, cols, ch = I1.shape
index = (enhance_thresh == 255)

replaced_index = np.zeros((rows, cols, ch), np.bool)
replaced_index[:, :, 0] = index
replaced_index[:, :, 1] = index
replaced_index[:, :, 2] = index

# replace only foreground indexes.
I2[replaced_index] = I2_foreground[replaced_index]
cv_iml.image_show(I2)