import matplotlib.pyplot as plt
import cv2
import numpy as np
from custom_classes import path


def image_show(img, cmap=None):
    """

    :param cmap:
    :param binary:
    :param img: img to display.
    :return: None
    """
    if cmap == 'gray':
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(img)
    plt.show()


def show_multiple_image_with(row=1, col=1):
    print("show  multiple image is running")


def removeBlackRegion(img):
    """
       This function remove the black region form image by cropping largest contours form the image.
       :param img: input image
       :return: image with removed black region but not removed black regions completely we need to apply some
       thresholding to rows and col to completely remove the black region.
       """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Find all contours form the gray image.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = -1
    best_cnt = None

    # Find contours with largest area.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    # Get coordinate of largest contours
    x, y, w, h = cv2.boundingRect(best_cnt)

    # Crop original with coordinate of largest contour.
    crop = img[y:y + h, x:x + w]
    return crop


def generate_patches_of_image(input_folder_path="", out_folder_path="", annotation_file_path=None, patch_size=0,
                              annotation_label=""):
    # get all images name form given folder.
    all_images_name = path.read_all_files_name_from(folder_path=input_folder_path, file_extension='.jpg')
    # if you want to use give annotation to file of not.
    if annotation_file_path:
        # open annotation file in which you put the annotation of generated image.
        f = open(annotation_file_path, "a")
    print("generating patches ...")
    for image_name in all_images_name:
        # reading image form given path.
        image = cv2.imread(input_folder_path + "/" + image_name)
        # get shape of images
        row, col, c = image.shape
        # if image size is less then images size then generate no patch.
        if row and col < patch_size:
            continue
        # get number of patches from given image.
        row_count = int(row / patch_size)
        col_count = int(col / patch_size)
        # handing row limit for patches loop
        for i in range(0, row_count * patch_size, patch_size):
            # handling col limit for patches
            for j in range(0, col_count * patch_size, patch_size):
                # pick specific patch form image.
                image_patch = image[i: i + patch_size, j: j + patch_size, :]
                # generate patch name with row and col name.
                patch_image_save_name = image_name[:-4] + "_" + str(i) + str(j) + ".jpg"
                # save the image and annotation in file.
                cv2.imwrite(out_folder_path + patch_image_save_name, image_patch)
                if annotation_file_path:
                    f.write(patch_image_save_name + " " + annotation_label)
                    f.write('\n')
    if annotation_file_path:
        f.close()
