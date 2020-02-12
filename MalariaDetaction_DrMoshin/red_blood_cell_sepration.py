import cv2
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from custom_classes import path

# Reading the image
path.init()

# file_name = path.dataset_path + "image_preview.png"
file_name = "/Users/qaziammar/Downloads/Microscope.JPG"
img = cv2.imread(file_name)
clone = img.copy()
# Converting the images into
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# kernel = np.ones((3, 3), np.uint8)
# closing = cv2.morphologyEx(im_bw, cv2.MORPH_CLOSE, kernel)
# dilation = cv2.dilate(im_bw, kernel, iterations=1)
# erosion = cv2.erode(dilation, kernel, iterations=1)
blur = cv2.GaussianBlur(src=gray,
                        ksize=(3, 3),
                        sigmaX=0)
edges = cv2.Canny(blur, 200, 255)
ret, thresh = cv2.threshold(edges, 225, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
print("length of contours is ", len(contours))
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

# create all-black mask image
mask = np.zeros(shape=img.shape, dtype="uint8")
# Draw rectangle on detected contours.
for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(img=mask, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=-1)

red_blood_cells = []

for c in contours:
    (x, y, w, h) = cv2.boundingRect(c)
    crop_image = clone[y:y + h, x: x + w]
    if w > 10 and h > 10:
        red_blood_cells.append(crop_image)

# image = cv2.bitwise_and(src1=img, src2=mask)
# plt.imshow(image, cmap='gray', vmin=0, vmax=255)
# plt.show()

i = 0
for cell in red_blood_cells:
    # Add path to the folder where you want to save your individual cells.
    outfile = '%s/%s.jpg' % ("/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/Results/Cell", i)
    cv2.imwrite(outfile, cell)
    i = i + 1

# https://datacarpentry.org/image-processing/09-contours/
