# This is a file where we fist test our code then implement it into other file
from custom_classes import path
import cv2
import psycopg2


def augment_image():
    images_folder_path = path.dataset_path + "blur_clear_foldscope/clear_patch"
    all_images_name = path.read_all_files_name_from(folder_path=images_folder_path, file_extension='.jpg')
    f = open(path.dataset_path + "blur_clear_foldscope/clear_patches_annotation.txt", "a")
    print("generating ...")
    for image_name in all_images_name:
        #         append images name in file "patches_locations.txt"
        counter = 0
        img = cv2.imread(images_folder_path + "/" + image_name)

        # get image height, width
        (h, w) = img.shape[:2]
        # calculate the center of the image
        center = (w / 2, h / 2)

        angle90 = 90
        angle180 = 180
        angle270 = 270

        scale = 1.0

        # Perform the counter clockwise rotation holding at the center
        # 90 degrees
        M = cv2.getRotationMatrix2D(center, angle90, scale)
        rotated90 = cv2.warpAffine(img, M, (h, w))
        patch_image_save_name = image_name[:-4] + "_" + str(angle90) + ".jpg"
        cv2.imwrite(path.dataset_path + "blur_clear_foldscope/clear_patch/" + patch_image_save_name, rotated90)
        f.write(patch_image_save_name + " " + "0")
        f.write('\n')

        # 180 degrees
        M = cv2.getRotationMatrix2D(center, angle180, scale)
        rotated180 = cv2.warpAffine(img, M, (w, h))
        patch_image_save_name = image_name[:-4] + "_" + str(angle180) + ".jpg"
        cv2.imwrite(path.dataset_path + "blur_clear_foldscope/clear_patch/" + patch_image_save_name, rotated180)
        f.write(patch_image_save_name + " " + "0")
        f.write('\n')

        # 270 degrees
        M = cv2.getRotationMatrix2D(center, angle270, scale)
        rotated270 = cv2.warpAffine(img, M, (h, w))
        patch_image_save_name = image_name[:-4] + "_" + str(angle270) + ".jpg"
        cv2.imwrite(path.dataset_path + "blur_clear_foldscope/clear_patch/" + patch_image_save_name, rotated270)
        f.write(patch_image_save_name + " " + "0")
        f.write('\n')

    print(counter)
    counter = counter + 1

    f.close()


def main():
    conn = psycopg2.connect("host=127.0.0.1 user=postgres password=12345678")

    # images_folder_path = path.dataset_path + "foldscope_dataset/test_patch"
    # all_images_name = path.read_all_files_name_from(folder_path=images_folder_path, file_extension='.jpg')
    # f = open(path.dataset_path + "foldscope_dataset/patches_annotatin_test.txt", "a")
    # print("generating ...")
    # for image_name in all_images_name:
    #     # append images name in file "patches_locations.txt"
    #     if 'c' in image_name:
    #         f.write(image_name + " " + "0")
    #     else:
    #         f.write(image_name + " " + "1")
    #
    #     f.write('\n')
    # f.close()




# %%


if __name__ == '__main__':
    main()

