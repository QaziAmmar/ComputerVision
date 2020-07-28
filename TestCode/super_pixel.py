from custom_classes import cv_iml, path


folder_path = path.dataset_path + "aug_test/"

cv_iml.augment_image(folder_path, '.jpg', rotation=True, flipping=True)