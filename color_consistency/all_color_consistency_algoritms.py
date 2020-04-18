import cv2
import numpy
from scipy.ndimage import label
from custom_classes import path, cv_iml
import colorcorrect.algorithm as cca

# Link of code:  https://pypi.org/project/colorcorrect/

# image_name = "4a_001.jpg"
dataset_path = path.dataset_path + "foldscope_sample/"
result_path = path.result_folder_path + "foldcsope_color_test/"
# dataset_path = path.dataset_path + image_name
all_images_name = path.read_all_files_name_from(dataset_path, ".jpg")
# all_images_name = all_images_name[420:len(all_images_name)]
# print(len(all_images_name))

# %%
# for single_image_name in all_images_name:
# imageName = "15b_003.jpg"
for single_image_name in all_images_name:
    print(single_image_name)
    img = cv2.imread(dataset_path + single_image_name)
    # cv2.imwrite(result_path + "stretch/" + "stretch_" + single_image_name,
    #             cca.stretch(img))
    cv2.imwrite(result_path + "stretch/" + "stretch_" + single_image_name, cca.stretch(img))
    cv2.imwrite(result_path + "grey_world/" + "grey_world_" + single_image_name, cca.grey_world(img))
    cv2.imwrite(result_path + "max_white/" + "max_white_" + single_image_name, cca.max_white(img))
    cv2.imwrite(result_path + "retinex/" + "retinex_" + single_image_name, cca.retinex(img))
    cv2.imwrite(result_path + "retinex_with_adjust/" + "retinex_with_adjust_" + single_image_name,
                cca.retinex_with_adjust(img))
    # cv2.imwrite(result_path + "standard_deviation_weighted_grey_world/" +
    #             "standard_deviation_weighted_grey_world_" + single_image_name,
    #             cca.standard_deviation_weighted_grey_world(img))

# cv_iml.image_show(img)
# cv_iml.image_show(cca.stretch(img))
# cv_iml.image_show(cca.grey_world(img))
# cv_iml.image_show(cca.max_white(img))
# cv_iml.image_show(cca.retinex(img))
# cv_iml.image_show(cca.retinex_with_adjust(img))
# cv_iml.image_show(cca.standard_deviation_weighted_grey_world(img))

#     save each image into its respective folder.
# cv2.imwrite(result_path + "stretch_" + imageName, cca.stretch(img))
# cv2.imwrite(result_path + "grey_world" + imageName, cca.grey_world(img))
# cv2.imwrite(result_path + "max_white" + imageName, cca.max_white(img))
# cv2.imwrite(result_path + "retinex" + imageName, cca.retinex(img))
# cv2.imwrite(result_path + "retinex_with_adjust" + imageName, cca.retinex_with_adjust(img))
# cv2.imwrite(result_path + "standard_deviation_weighted_grey_world" + imageName,
# cca.standard_deviation_weighted_grey_world(img))


# cv2.imwrite(result_path + "stretch/" + single_image_name, cca.stretch(img))
# cv2.imwrite(result_path + "grey_world/" + single_image_name, cca.grey_world(img))
# cv2.imwrite(result_path + "max_white/" + single_image_name, cca.max_white(img))
# cv2.imwrite(result_path + "retinex/" + single_image_name, cca.retinex(img))
# cv2.imwrite(result_path + "retinex_with_adjust/" + single_image_name, cca.retinex_with_adjust(img))
# cv2.imwrite(result_path + "standard_deviation_weighted_grey_world/" + single_image_name,
# cca.standard_deviation_weighted_grey_world(img))

# standard deviation weighted grey world
#          colorcorrect.algorithm.standard_deviation_weighted_grey_world
#          usage: image,subblock width(default:20), subblock height(default:20)
# standard deviation and luminance weighted gray world
#          colorcorrect.algorithm.standard_deviation_and_luminance_weighted_gray_world
#          usage: image,subblock width(default:20), subblock height(default:20)
# luminance weighted gray world
#           colorcorrect.algorithm.luminance_weighted_gray_world
#          usage: image,subblock width(default:20), subblock height(default:20)
# automatic color equalization
#           colorcorrect.algorithm.automatic_color_equalization
#           usage: image,slope(default:10),limit(default:1000)
