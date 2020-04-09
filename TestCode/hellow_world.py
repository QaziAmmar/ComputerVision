import cv2
import numpy
from scipy.ndimage import label
from custom_classes import path, cv_iml
import colorcorrect.algorithm as cca


from_folder_path = path.dataset_path + "Malaria_dataset/malaria/"

required_images = ['2a_001.jpg', '2a_014.jpg', '2a_036.jpg', '2a_038.jpg', '2a_072.jpg',
                   '4a_002.jpg', '4a_045.jpg', '4a_093.jpg', '6a_001.jpg', '7b_004.jpg',
                   '15b_001.jpg', 'a70.jpg', 'a80.jpg', 'a94.jpg', 'w31_001.jpg', 'w64_007.jpg',
                   'w66_101.jpg']

save_folder_path = path.dataset_path + "count_tester/"

for imageName in required_images:
    img = cv2.imread(from_folder_path + imageName)
    cv2.imwrite(save_folder_path + "stretch/" + "stretch_" + imageName, cca.stretch(img))
    cv2.imwrite(save_folder_path + "grey_world/" + "gw_" + imageName, cca.grey_world(img))
    cv2.imwrite(save_folder_path + "max_white/" + "mw_" + imageName, cca.max_white(img))
    cv2.imwrite(save_folder_path + "retinex/" + "ret_" + imageName, cca.retinex(img))
    cv2.imwrite(save_folder_path + "retinex_with_adjust/" + "rwa_" + imageName, cca.retinex_with_adjust(img))
    cv2.imwrite(save_folder_path + "automatic_color_equalization/" + "acq_" + imageName, cca.automatic_color_equalization(img))

