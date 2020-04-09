import cv2
import itertools
from custom_classes import path

# this Reading annotation file of dataset.
# # %%

dataset_path = path.dataset_path + "Malaria_dataset/malaria/"
annotation_path = path.dataset_path + "Malaria_dataset/malaria.txt"
save_folder_path = path.result_folder_path + "enhanceImages/"

images_name = path.read_all_files_name_from(dataset_path, '.jpg')
annotation_file = open(annotation_path, 'r')
lines = annotation_file.readlines()


# %%
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

# %%
# Group list by image name

key_f = lambda x: x[0]

for key, group in itertools.groupby(image_annotation, key_f):

    img_name = key + ".jpg"
    img = cv2.imread(dataset_path + img_name)
    single_image_annotation_list = list(group)
    for image in single_image_annotation_list:
        # img_name = image[0] + ".jpg"
        type = image[1]
        point1 = image[2]
        point2 = image[3]
        point3 = image[4]

        center_coordinate1 = (int(point1[0]), int(point1[1]))
        center_coordinate2 = (int(point2[0]), int(point2[1]))
        center_coordinate3 = (int(point3[0]), int(point3[1]))
        radius = 2
        color = (255, 0, 0)
        thickness = 2

        img = cv2.circle(img, center_coordinate1, radius, color, thickness)
        img = cv2.circle(img, center_coordinate2, radius, color, thickness)
        img = cv2.circle(img, center_coordinate3, radius, color, thickness)

    cv2.imwrite(save_folder_path + img_name, img)




