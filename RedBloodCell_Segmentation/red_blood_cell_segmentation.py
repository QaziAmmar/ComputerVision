#  Red Blood segmentation.
# DataSet: Complete-Blood-Cell-Count-Dataset-master
import path
from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

path.init()


# Defining Classes
class Cell:
    cell_name = "empty"
    xmin = ""
    ymin = ""
    xmax = ""
    ymax = ""


# Defining directory structures/names.
dataset_name = "Complete-Blood-Cell-Count-Dataset-master/"
testing_path = "Testing/"
training_path = "Training/"
validation_path = "Validation/"
images = "Images/"
annotation = "Annotations/"
save_results_folder = path.result_folder_path + "RGB_Annotated"

cell_dataset_path = path.dataset_path + dataset_name
training_images = cell_dataset_path + training_path + images
training_images_annotation = cell_dataset_path + training_path + annotation

# %%
# Reading all annotations files.
annotation_files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(training_images_annotation):
    for file in f:
        if '.xml' in file:
            annotation_files.append(os.path.join(r, file))
# %%
# Reading all images form file.
all_images_name = []
# r=root, d=directories, f = files
for r, d, f in os.walk(training_images):
    for file in f:
        if '.jpg' in file:
            all_images_name.append(file)


# %%
# Parsing XML.
def annotation_xml_form(file_path=""):
    all_cells_in_file = []
    tree = ET.parse(file_path)
    root = tree.getroot()

    for cell_name in root.findall('object'):
        temp_cell_coordinate = Cell()

        temp_cell_coordinate.cell_name = cell_name.find('name').text
        bndbox = cell_name.find('bndbox')

        temp_cell_coordinate.xmin = bndbox.find('xmin').text
        temp_cell_coordinate.ymin = bndbox.find('ymin').text
        temp_cell_coordinate.xmax = bndbox.find('xmax').text
        temp_cell_coordinate.ymax = bndbox.find('ymax').text
        all_cells_in_file.append(temp_cell_coordinate)

    return all_cells_in_file


def change_extension(filename="", new_extension="xml"):
    filename = filename[:-3]
    filename = filename + new_extension
    return filename


# %%
i = 0
# testing annotations plot on cell images.

for image_name in all_images_name:

    annotation_name = change_extension(image_name)
    image_path = training_images + image_name
    annotation_path = training_images_annotation + annotation_name

    image = cv2.imread(image_path)
    image_annotation = annotation_xml_form(file_path=annotation_path)

    for cells in image_annotation:
        xmin = int(cells.xmin)
        ymin = int(cells.ymin)
        xmax = int(cells.xmax)
        ymax = int(cells.ymax)
        # cv2.drawContours(first_image, (xmin, ymin, xmax, ymax), -1, (0, 255, 0), 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
    #       save the annotated images.
    outfile = '%s/%s.jpg' % (save_results_folder, i)
    cv2.imwrite(outfile, image)
    i = i + 1


# %%
# applying k-means clustering algorithm.

def apply_kmeans_clustring(image_path=""):
    pic = cv2.imread(image_path) / 255
    pic_n = pic.reshape(pic.shape[0] * pic.shape[1], pic.shape[2])
    pic_n.shape
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pic_n)
    pic2show = kmeans.cluster_centers_[kmeans.labels_]
    cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
    plt.imshow(cluster_pic, cmap='gray', vmin=0, vmax=255)
    plt.show()


def apply_morophological_operation(image_path=""):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Noise removal using Morphological
    # closing operation
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               kernel, iterations=2)

    # Background area using Dialation
    bg = cv2.dilate(closing, kernel, iterations=1)

    # Finding foreground area
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
    ret, fg = cv2.threshold(dist_transform, 0.02
                            * dist_transform.max(), 255, 0)

    plt.imshow(fg, cmap='gray', vmin=0, vmax=225)
    plt.show()


# apply_kmeans_clustring(image_path)
apply_morophological_operation(image_path)
