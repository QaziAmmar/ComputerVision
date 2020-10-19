# This is a testing code for rgb segmentation
# Code Link: https://github.com/LumRamabaja/Red_Blood_Cell_Segmentation
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


def image_thresh_with_divide(image, num_of_parts):
    # This function get gray image divide it into sub parts, apply otsu thresh on these images part
    # and return binary mask.
    # divide image into 10 equal parts.
    img_clone = image
    image_parts = num_of_parts

    r, c = darker.shape
    r_step = int(r / image_parts)
    c_step = int(c / image_parts)

    otsu_binary_mask = np.zeros((r, c), np.uint8)

    #  divide image into 10 small parts and then compute background and foreground threshold for these
    #  image and then combine these images.
    if r % image_parts == 0:
        row_range = (r - r_step) + 1
    else:
        row_range = (r - r_step)

    if c % image_parts == 0:
        col_range = (c - c_step) + 1
    else:
        col_range = (c - c_step)

    for temp_row in range(0, row_range, r_step):
        for temp_col in range(0, col_range, c_step):
            # separating image part for complete image.
            temp_imge = img_clone[temp_row: temp_row + r_step, temp_col: temp_col + c_step]
            # Otsu's threshold
            ret, thresh = cv2.threshold(temp_imge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # invert
            invert_thresh = cv2.bitwise_not(thresh)
            # combining image.
            otsu_binary_mask[temp_row: temp_row + r_step, temp_col: temp_col + c_step] = invert_thresh

    return otsu_binary_mask


def watershed_labels(binary_mask):
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(binary_mask)
    localMax = peak_local_max(D, indices=False, min_distance=10,
                              labels=binary_mask)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=binary_mask)

    return labels


def plot_labels_on(annotated_img, forground_background_mask, labels, cell_count):
    clotted_cell_image = forground_background_mask.copy()
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
        if (w < 8 or h < 8) or (w > 30 or h > 30):
            continue
        cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        roi = image[y:y + h, x:x + w]
        clotted_cell_image[y:y + h, x:x + w] = 0
        cell_count += 1
        # cv2.imwrite(directory + "/cell{}.png".format(num), roi)
        num = num + 1
        # save an image where every cell is segmented by rectangles.
    # cv2.imwrite(directory + "segmented_" + image_name, img_copy)
    # cv2.imwrite(mask_directory + single_image_name, labels)

    clotted_cell_image = clotted_cell_image * 255

    return annotated_img, clotted_cell_image, cell_count


def is_new_cell_segments_found(new_count, pre_count):
    # this function terminates the loop when new generated cell are less than 10
    new_cells = new_count - pre_count
    if new_cells > 10:
        return True
    else:
        return False
#########################################################################
directory = path.dataset_path + "foldscope_test/foldsocpe_all_sample/original/"
all_images_name = path.read_all_files_name_from(directory, '.jpg')
image_name = "20200108_171011.jpg"

image = cv2.imread(directory + image_name)

total_cell_count = 0
# image = cv2.pyrMeanShiftFiltering(image, 10, 10)
annotated_img = image.copy()
# convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# improve the contrast of our images
darker = cv2.equalizeHist(gray)

forground_background_mask = image_thresh_with_divide(darker, 8)

# %%
# find contours of newimg
contours, hierarchy = cv2.findContours(forground_background_mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

# %%
# filling the "holes" of the cells
for cnt in contours:
    cv2.drawContours(forground_background_mask, [cnt], 0, 255, -1)

# %%
# get labels with watershed algorithms.
labels = watershed_labels(forground_background_mask)

# %%
# plot annotation on image
annotated_img, clotted_cell_image, total_cell_count = plot_labels_on(annotated_img, forground_background_mask,
                                                                     labels, total_cell_count)

# %%
# start testing code.
kernel = np.ones((3, 3), np.uint8)
counter = 0


while counter < 10:
    # Apply erosion on the image so that large cell can also be seprated
    clotted_cell_image = cv2.resize(clotted_cell_image, image.shape[1::-1])
    remaining_image = cv2.bitwise_and(image, image, mask=clotted_cell_image)
    remaining_image_sharp = cv_iml.apply_sharpening_on(remaining_image)

    gray = cv2.cvtColor(remaining_image_sharp, cv2.COLOR_BGR2GRAY)
    # improve the contrast of our images
    darker = cv2.equalizeHist(gray)

    ret, thresh = cv2.threshold(darker, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    forground_background_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=kernel)

    labels = watershed(forground_background_mask)
    prev_count = total_cell_count

    annotated_img, clotted_cell_image, total_cell_count = plot_labels_on(annotated_img, forground_background_mask,
                                                                         labels, total_cell_count)
    counter += 1
    # this condition terminates the loop if no new cells are found.
    print(total_cell_count)

    if is_new_cell_segments_found(total_cell_count, prev_count):
        continue
    else:
        print("code is terminated because no new cells found")
        break

print(counter)