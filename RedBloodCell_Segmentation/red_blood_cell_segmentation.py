#  Red Blood segmentation.
# DataSet: Complete-Blood-Cell-Count-Dataset-master
import path
from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import cv2
import matplotlib.pyplot as plt

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
    outfile = '%s/%s.jpg' % ("/Users/qaziammar/Downloads/RGB_Annotated", i)
    cv2.imwrite(outfile, image)
    i = i + 1

