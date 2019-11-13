import matplotlib.pyplot as plt
import path
import cv2
import imutils
import time
import os

path.init()


# save stitched images in folder.
def remove_black_region(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = -1
    best_cnt = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    x, y, w, h = cv2.boundingRect(best_cnt)
    crop = img[y:y + h, x:x + w]
    return crop


def read_all_images_name(folder_path):
    """

    :param folder_path: this is the path of folder in which we pick all images.
    :return: this function return sorted name of images with complete path
    """
    # Reading all images form file.
    all_images_name = []
    images_names = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(folder_path):
        for file in f:
            if '.JPG' in file:
                all_images_name.append(r + file)
                images_names.append(file)
    return sorted(all_images_name), sorted(images_names)


def remove_100_row_of(image):
    row, col, ch = image.shape
    cropped_image = image[250: row - 250, 250: col - 250, :]
    return cropped_image


def read_all_images_form(images_names, file_name):
    # Reading all images form file.
    all_images_array = []
    for image_name, file in zip(images_names, file_name):
        img = cv2.imread(image_name)
        img = remove_black_region(img)
        # save the image with same name
        # img = remove_100_row_of(img)
        cv2.imwrite(file, img=img)


results_folder = path.dataset_path + "Malaria_Dataset_self/SHIF_images/miscrscope_panaroma.2/"

# This function perform stitching on images.
images_name, file_name = read_all_images_name(results_folder)
read_all_images_form(images_name, file_name)
