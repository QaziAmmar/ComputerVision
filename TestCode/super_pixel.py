from custom_classes import cv_iml, path
import os

for root, dirs, files in os.walk(path.dataset_path):
    for dirs_name in root:
        print(dirs_name)
