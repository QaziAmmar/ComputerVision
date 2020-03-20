# This is a file where we fist test our code then implement it into other file
# Annotation draw testing code.
from custom_classes import path, cv_iml
import re
import os
import cv2

annotation_path = path.dataset_path + "Malaria_dataset/malaria.txt"
images_path = path.dataset_path + "Malaria_dataset/malaria/"


#  parse annotation file.
def parse_annotation_file(file_path):
    image_annotation = []
    f = open(file_path, "r", encoding='utf-8')
    for line in f:
        # Parsing Annotation file.
        line_split_by_bracket = line.split('[')

        image_name = line_split_by_bracket[0].split('.')[1]
        image_name = image_name.split('=')[1]

        type = line_split_by_bracket[0].split('.')[2]
        type = type.split('=')[1]

        point1 = line_split_by_bracket[1][:-2].split(',')
        point2 = line_split_by_bracket[2][:-2].split(',')
        point3 = line_split_by_bracket[3][:-3].split(',')

        image_annotation.append((image_name, type, point1, point2, point3))
    f.close()
    return image_annotation


# Get image annotation of file
image_annotation = parse_annotation_file(annotation_path)

for image in image_annotation:
    img_name = image[0] + ".jpg"
    type = image[1]
    point1 = image[2]
    point2 = image[3]
    point3 = image[4]
    img = cv2.imread(images_path + img_name)
    img = cv2.rectangle(img, (int(point2[0]), int(point2[1])),
                        (int(point3[0]), int(point3[1])), (0, 255, 0), 1)
    cv_iml.image_show(img)


