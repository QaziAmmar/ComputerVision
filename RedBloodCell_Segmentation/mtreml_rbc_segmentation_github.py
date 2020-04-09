# Standard imports
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
import os
from custom_classes import path, cv_iml

# Link: https://github.com/mtreml/rbc-segmentation/blob/master/rbc_segmentation_v1.ipynb

folder_path = "count_tester/retinex_with_adjust/"
image_path = os.path.join(path.dataset_path, folder_path)
result_savedirectory = path.result_folder_path + "count_tester/mtremel/retinex_with_adjust/"

text_file_path = os.path.join(result_savedirectory, "count.txt")
file = open(text_file_path, "a")


all_images_name = path.read_all_files_name_from(image_path, '.jpg')

for image_name in all_images_name:

    # Read image
    image = cv2.imread(image_path + image_name)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv_iml.image_show(gray, 'gray')
    # Image.fromarray(gray)

    # equalizing the image
    darker = cv2.equalizeHist(gray)
    # cv_iml.image_show(darker, 'gray')

    # threshold
    ret, thresh = cv2.threshold(darker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv_iml.image_show(thresh, 'gray')

    # invert
    newimg = cv2.bitwise_not(thresh)
    # cv_iml.image_show(newimg, 'gray')

    # opening
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(newimg, cv2.MORPH_OPEN, kernel)
    # cv_iml.image_show(opened, 'gray')

    # find contours of white objects
    im2, contours, hierarchy = cv2.findContours(opened, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        cv2.drawContours(opened, [cnt], 0, 255, -1)


    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(opened)
    localMax = peak_local_max(D, indices=False, min_distance=15,
                              labels=opened)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=opened)
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))


    # loop over the unique labels returned by the Watershed
    # algorithm
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue

        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # draw a circle enclosing the object
        # ((x, y), r) = cv2.minEnclosingCircle(c)
        x, y, w, h = cv2.boundingRect(c)
        # cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 2)
        # cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    file.write(image_name + "  " + str(len(np.unique(labels)) - 1) + "\n")
    cv2.imwrite(result_savedirectory + image_name, image)
    # cv_iml.image_show(image)
file.close()
#%%
# %matplotlib inline
