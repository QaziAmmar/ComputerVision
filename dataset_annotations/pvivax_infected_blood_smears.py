from custom_classes import path, cv_iml
import json
import cv2

from dataset_annotations.pvivax_model import Annotation_Model

healthy_path = path.dataset_path + "malaria/"
healthy_train_annotation_path = healthy_path + "training.json"
healthy_test_annotation_path = healthy_path + "test.json"

infected_folder_path = healthy_path + "malaria/"
infected_train_annotation_path = infected_folder_path + "training.json"
infected_test_annotation_path = infected_folder_path + "test.json"

with open(healthy_train_annotation_path) as healthy_train_annotation_file:
    healthy_train_annotation = json.load(healthy_train_annotation_file)

# %%

python_annotataiton = []

# annotation = healthy_train_annotation[1]
for annotation in healthy_train_annotation:
    python_annotataiton.append(Annotation_Model(annotation))

# %%

test_plot = python_annotataiton[50]
img = cv2.imread(healthy_path + test_plot.image.path_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

for object in test_plot.objects:
    pt1 = (object.bounding_box.x1, object.bounding_box.y1)
    pt2 = (object.bounding_box.x2, object.bounding_box.y2)
    img = cv2.rectangle(img, pt2, pt1, (36, 255, 12), 2)
    # cv2.putText(img, object.category, (int(pt1[0]), int(pt1[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

cv_iml.image_show(img)
