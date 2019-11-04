import matplotlib.pyplot as plt
import path
import cv2

path.init()

folder_name = path.dataset_path + "Malaria_Dataset_self/SHIF_images/"
complete_image = folder_name + "IMG_4030.JPG"
crop_image = folder_name + "croped_image.JPG"
unBlur_image = folder_name + "unBlur_image.JPG"

img = cv2.imread(folder_name + "1_blur_map.JPG")

plt.imshow(img, cmap='gray', vmax=255, vmin=0)
plt.show()
