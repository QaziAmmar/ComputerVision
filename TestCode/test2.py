
from RedBloodCell_Segmentation.seg_dr_waqas_watershed_microscope_single_image import get_detected_segmentaion
from custom_classes import cv_iml

image_path = "/home/iml/Desktop/qazi/Model_Result_Dataset/Dataset/IML_loclization_final/p.v/100X_crop/IMG_4471.JPG"
annotated_img, individual_cell_images, json_object = get_detected_segmentaion(image_path)
cv_iml.image_show(annotated_img)