# This is a testing code for rgb segmentation
#Code Link: https://github.com/LumRamabaja/Red_Blood_Cell_Segmentation
# Helping Links:
# https://www.pyimagesearch.com/2015/11/02/watershed-opencv/
# Also this author has classification code.
# This code required python 3.7.
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from custom_classes import path, cv_iml

#########################################################################
directory = path.result_folder_path + "microscope_test/sample_images/"
save_folder_name = path.result_folder_path + "microscope_test/lumrama/"
all_images_name = path.read_all_files_name_from(directory, '.JPG')
# image_name = "IMG_4542.JPG"
for image_name in all_images_name:
    print(image_name)
    image = cv2.imread(directory + image_name)
    img_copy = image.copy()

    # convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # improve the contrast of our images
    darker = cv2.equalizeHist(gray)
    # Otsu's threshod
    ret, thresh = cv2.threshold(darker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # invert
    newimg = cv2.bitwise_not(thresh)

    #%% Morphological opening
    # kernel = np.ones((2, 2), np.uint8)
    # opening = cv2.morphologyEx(newimg, cv2.MORPH_OPEN, kernel)
    # cv_iml.image_show(opening, 'gray')

    #%%
    # find contours of newimg
    contours, hierarchy = cv2.findContours(newimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))

    #%%
    # filling the "holes" of the cells
    for cnt in contours:
        cv2.drawContours(newimg, [cnt], 0, 255, -1)

    #%%
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(newimg)
    localMax = peak_local_max(D, indices=False, min_distance=20,
                              labels=newimg)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=newimg)

    # loop over the unique labels returned by the Watershed algorithm
    num = 0
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
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

        # saving every single cell as a rectangular image.
        x, y, w, h = cv2.boundingRect(c)
        if (w < 80 or h < 80) or (w > 150 or h > 150):
            continue
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 0, 0), 2)
        roi = image[y:y + h, x:x + w]
        # cv2.imwrite(directory + "/cell{}.png".format(num), roi)
        num = num + 1

    # save an image where every cell is segmented by rectangles.
    cv2.imwrite(save_folder_name + image_name, img_copy)