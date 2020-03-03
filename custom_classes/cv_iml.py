import matplotlib.pyplot as plt
import cv2
import numpy as np
from custom_classes import path


def image_show(img, cmap=None, suptitle=""):
    """

    :param cmap:
    :param binary:
    :param img: img to display.
    :return: None
    """
    fig = plt.figure()
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')

    # ax = fig.add_subplot(111)
    # fig.subplots_adjust(top=0.85)
    # ax.set_title('axes title')
    #
    # ax.set_xlabel('xlabel')
    # ax.set_ylabel('ylabel')
    #
    # ax.text(3, 8, 'boxed italics text in data coords', style='italic',
    #         bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})
    #
    # ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)
    #
    # ax.text(3, 2, 'unicode: Institut für Festkörperphysik')
    #
    # ax.text(0.95, 0.01, 'colored text in axes coords',
    #         verticalalignment='bottom', horizontalalignment='right',
    #         transform=ax.transAxes,
    #         color='green', fontsize=15)
    #
    # ax.plot([2], [1], 'o')
    # ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
    #             arrowprops=dict(facecolor='black', shrink=0.05))
    #
    # ax.axis([0, 10, 0, 10])

    if cmap == 'gray':
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    else:
        plt.imshow(img)
    plt.show()


def apply_sharpening_on(image):
    # Create kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # Sharpen image
    sharp_image = cv2.filter2D(image, -1, kernel)
    return sharp_image


def show_multiple_image_with(row=1, col=1, images=[], titles=[]):
    """
    This not stable function.
    :param row:
    :param col:
    :param images:
    :param titles:
    :return:
    """
    fig, ax = plt.subplots(row, col, figsize=(10, 10), sharex=True, sharey=True)
    image_counter = 0
    for i in range(row):
        for j in range(col):
            ax[i, j].imshow(images[image_counter])
            ax[i, j].set_title(titles[image_counter])

    for a in ax.ravel():
        a.set_axis_off()

    plt.tight_layout()
    plt.show()


def removeBlackRegion(img):
    """
    This is a stable function.
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


def get_image_patches_by_sliding_window(img, stepSize, window_size, overlapping):
    """
    -> This is not stable function.
    -> Need to through exception when overlapping is 100%
    -> This function take a full image and make patches of that image save them into array and
        return that arrray.
    :param img: full image form where you want to get patches.
    :param stepSize:
    :param window_size:
    :param overlapping: how much overlapping you need in you images.
    :return: array of images extracted by sliding window.
    """
    # read the image and define the stepSize and window size
    # (width,height)
    if overlapping == 100:
        return None
    # generation step size for overlapping
    overlapping = 100 - overlapping
    stepSize = int(stepSize * (overlapping / 100))

    patches = []
    image = img  # your image path
    tmp = image  # for drawing a rectangle
    (w_width, w_height) = (window_size, window_size)  # window size
    for x in range(0, image.shape[1] - w_width, stepSize):
        for y in range(0, image.shape[0] - w_height, stepSize):
            window = image[x:x + w_width, y:y + w_height, :]
            # add window into your patches array.
            patches.append(window)

    return patches
