import cv2
import numpy as np
from custom_classes import path
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


def image_show(img, cmap=None, suptitle=""):
    """
    Version = 1.1
    This is a stable function.
    This function take an image and how it by using matplot function.
    :param cmap: 'Reds', 'Greens', 'Blues'
    :param binary:
    :param img: img to display.
    :return: None
    """
    fig = plt.figure()
    fig.suptitle(suptitle, fontsize=14, fontweight='bold')

    if cmap is not None:
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
    else:
        plt.imshow(img)
    plt.show()


def show_histogram(img, title=None):
    """
    version: 1.1
    This is a stable function.
    This function shows the histogram of image. It first check if image is gray scale or rgb and then
    plot histogram according to it
    :param img: input image of which histogram is need to plot
    :param title: title of plot
    :return: None
    """
    if len(img.shape) < 3:
        histr = cv2.calcHist([img], [0], None, [256], [0, 256])

        plt.plot(histr, 'black', label='gray')
        plt.title("Grayscale Image Histogram")
    elif len(img.shape) == 3:
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histr, color=col, label=col)
            plt.xlim([0, 256])
            plt.title("RGB Image Histogram")
    plt.legend()
    if title is not None:
        plt.title(title)
    plt.show()


def apply_sharpening_on(image):
    """
    Version = 1.0
    This is a stable function.
    Apply sharpening kernel on image.
    :param image:
    :return:
    """
    # Create kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # Sharpen image
    sharp_image = cv2.filter2D(image, -1, kernel)
    return sharp_image


def show_multiple_image_with(row=1, col=1, images=[], titles=[]):
    """
    Version = 1.0
    This not stable function.
    We want to show multiple images side by side using this function.
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
    This function remove the black region form image by cropping the largest contours form the image.
    use this function to remove the black region in microscope images, that may cause problems.
    Version = 1.0
    # Link: ?
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
        return that array.
        # Link: ?
    Version = 1.0
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
            # add the window into your patches array.
            patches.append(window)

    return patches


# demonstration of calculating metrics for a neural network model using sklearn

def get_f1_score(actual_labels, preds_labels, pos_label=1, plot_confusion_matrix=False):
    """
    Calculate the F1 score of CNN prediction.
    This method works for both binary class and multiclass. For binary class you have to
    mention pos_label and for multiclass pass pos_label =1
    # Link: ?
    Version = 1.1
    :param actual_labels: labels of testing images. Y_label
    :param preds_labels: predicted labels by trained CNN
    :param pos_label: The positive label for calculating the precision and recall. For
    the multiclass problem this label is set to 1.
    :return: this function return confusion matrix.
    """
    # demonstration of calculating metrics for a neural network model using sklearn
    if pos_label == 1:
        # For multiclass classification.
        accuracy = accuracy_score(actual_labels, preds_labels)
        precision = precision_score(actual_labels, preds_labels, pos_label=pos_label, average="weighted")
        recall = recall_score(actual_labels, preds_labels, pos_label=pos_label, average="weighted")
        f1 = f1_score(actual_labels, preds_labels, pos_label=pos_label, average="weighted")
        print('Accuracy: %f' % accuracy)
        print('Precision: %f' % precision)
        print('Recall: %f' % recall)
        print('F1 score: %f' % f1)
    else:
        accuracy = accuracy_score(actual_labels, preds_labels)
        print('Accuracy: %f' % accuracy)
        # # precision tp / (tp + fp)
        precision = precision_score(actual_labels, preds_labels, pos_label=pos_label)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(actual_labels, preds_labels, pos_label=pos_label)
        print('Recall: %f' % recall)
        # f1: 2 tp / (2 tp + fp + fn)
        f1 = f1_score(actual_labels, preds_labels, pos_label=pos_label)
        print('F1 score: %f' % f1)
    # ROC AUC
    # auc = roc_auc_score(test_labels, basic_cnn_preds_labels)
    # print('ROC AUC: %f' % auc)
    # confusion matrix
    matrix = confusion_matrix(actual_labels, preds_labels)
    print(matrix)
    if plot_confusion_matrix:
        show_confusion_matrix(matrix)


def show_confusion_matrix(matrix, labels: None):
    """
    This function plot the confusion matrix on.
    Version = 1.1
    :param matrix: matrix which we have to show
    :param labels: labels of classes
    :return: None
    """
    cm = matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    # ax.set_xticklabels([''] + labels)
    # ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# color_constancy code Start.
# Version 1.0
# This is a simple color balancing algorithm that balance color in the image.
# Link: https://gist.github.com/DavidYKay/9dad6c4ab0d8d7dbf3dc

def apply_mask(matrix, mask, fill_value):
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    return masked.filled()


def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


def color_constancy(img, percent=1):
    """

    :param img:
    :param percent:
    :return: return image with consistent colors.
    """
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv2.split(img):
        cumhist = np.cumsum(cv2.calcHist([channel], [0], None, [256], (0, 256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv2.LUT(channel, lut.astype('uint8')))
    return cv2.merge(out_channels)

# color_constancy code End.
