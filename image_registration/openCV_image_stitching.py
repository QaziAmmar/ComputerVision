import matplotlib.pyplot as plt
import path
import cv2
import imutils
import time
import numpy as np
import os

# from optical_flow import IML_optical_flow
# This current working code of images stitching.

path.init()

# only required varibale to run the code is.
# images_folder_path = "path of datafolder"

__author__ = "Qazi Ammar Arshad"
__email__ = "qaziammar.g@gmail.com"
__status__ = "This is working code of python that stitched, images automatically. Link of code: " \
             "https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/ "
__description__ = "This Code stitch both video frames and multiple image form same folder and save " \
                  "stitched image into current folder " \
                  "Code has 2 Selection Mode" \
                  "If selection == 1 then it stitch, images form same folder." \
                  "If selection == 2 then it stitch, video frames of given video" \

def iml_imshow(img):
    """

    :param img: img to display.
    :return: None
    """
    plt.imshow(img)
    plt.show()


def image_sharpning(image):
    """

    :param image: input image which is convolved with kernel_sharpening
    :return: sharped images
    """
    # Create  shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    return sharpened


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    This function resize the image, while maintaining it aspectt ratio You only need to give only one
     parameter height or width and image is resized accordingly.
    :param image: input image.
    :param width: optional.
    :param inter: optional
    :return: Resized image which has same aspect ratio as original image.
    """
    # Initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def remove_black_region(img):
    """
    This function remove the black region form image by cropping largest contours form the image.
    :param img: input images
    :return: image with removed black region but not removed black regions completely we need to apply some
    thresholding to rows and col to completly remove the black region.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Find all contours form the gray image.
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = -1
    best_cnt = None

    # Find contours with largest area.
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt

    # Get coordinate of largest contours
    x, y, w, h = cv2.boundingRect(best_cnt)

    # Crop original with coordinate of largest contour.
    crop = img[y:y + h, x:x + w]
    return crop


def read_all_images_name(folder_path):
    """

    :param folder_path: this is the path of folder from which we pick all images.
    :return: this function return sorted name of images with complete path
    """
    # Reading all images form file.
    all_images_name = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(folder_path):
        for file in f:
            if '.jpg' in file:
                all_images_name.append(r + file)
    return sorted(all_images_name)


def define_row_col_crop_limit(img, percentage_limit=6):
    """
    :param img: Input image.
    :param percentage_limit: This is the limit of number_of_zero_counter_in_single_row/col_of_image.
    :return: crop point of image to fully remove black region.
    """
    # Convert image RGB to Grayscale.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # Get total rows and cols of image.
    row, col = gray.shape

    # Crop point of Image eg. row_limit = 200 (crop top and bottom row/col or image.)
    row_limit = 0
    col_limit = 0

    # Set loop (jump = 10) to speed up the loop.
    for i in range(0, row, 10):
        temp_img_row = gray[i, :]

        # Count the number of pixels that has less value then 10 (consider as black pixel.)
        zero_count = np.sum(temp_img_row < 10)
        # Count the number of non_zero pixel in current image row.
        non_zero_count = col - zero_count

        # Percentage of zero_pixel in a col.
        percentage = (zero_count / col) * 100
        # If percentage reaches our defined threshold then terminate the loop.
        if percentage < percentage_limit:
            row_limit = i
            break

        # Set loop (jump = 10) to speed up the loop.
        for j in range(0, col, 10):
            temp_img_col = gray[:, j]

            # Count the number of pixels that has less value then 10 (consider as black pixel.)
            zero_count = np.sum(temp_img_col < 10)
            # Count the number of non_zero pixel in current image col.
            non_zero_count = col - zero_count

            # Percentage of zero_pixel in a row
            percentage = (zero_count / row) * 100
            # If percentage reaches our defined threshold then terminate the loop.
            if percentage < percentage_limit:
                col_limit = j
                break

    # Return number of rows and cols to cropped.
    return row_limit, col_limit


def remove_non_zero_row_col_of(image):
    """

    This function take image, first find crop point of row and col of image and then crop image from
    calculated row and col point/limit.
    :param image: input image.
    :return: image that cropped and have no black region.
    """
    # Find cut points of image.
    row_limit, col_limit = define_row_col_crop_limit(image)
    row, col, ch = image.shape

    # Crop image form calculated point of row and col.
    cropped_image = image[int(row_limit * 0.6): row - int(row_limit * 0.6),
                    int(col_limit * 0.5): col - int(col_limit * 0.5), :]
    # Return cropped Image..
    return cropped_image


def read_all_images_form(images_names):
    """
    This function load images for given path and apply needed methods (sharpening, black pixel removal etc)
    and make a images_array.

    :param images_names: absolute path of all images to be load.
    :return: array of loaded images form give path array = images_names
    """

    all_images_array = []
    # Loop through all images.
    for image_name in images_names:
        # Read Image form given path.
        img = cv2.imread(image_name)
        # Uncomment the required method to apply.
        # img = remove_black_region(img)
        # img = remove_non_zero_row_col_of(img)
        # img = image_sharpning(img)
        all_images_array.append(img)

    return all_images_array


def extract_key_frames_from_movie(movie_path):
    # Load vidoe form given path.
    cap = cv2.VideoCapture(movie_path)
    # Count Number of Frames form video.
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # This Array contains only selected frame of video.
    key_frames = []

    # Read video frame by jump of 3.
    for i in range(0, int(frame_count), 10):
        # Lode i frame form video.
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = cap.read()

        # Comment/Uncomment the required method to apply.
        # frame = cv2.transpose(frame)
        # frame = image_resize(frame, height=300)
        frame = remove_black_region(frame)
        frame = remove_non_zero_row_col_of(frame)

        key_frames.append(frame)
        cv2.imwrite("/Users/qaziammar/Documents/frames/" + str(i) + ".jpg", frame)
    return key_frames


def stitch_image(img1, img2):
    key_frames = [img1, img2]
    stitcher = cv2.Stitcher.create(mode=1)
    (status, stitched) = stitcher.stitch(key_frames)
    if status == 1:
        print(" ERR_NEED_MORE_IMGS = 1,")
    elif status == 2:
        print("ERR_HOMOGRAPHY_EST_FAIL = 2,")
    elif status == 3:
        print("ERR_CAMERA_PARAMS_ADJUST_FAIL = 3")
    elif status == 0:
        print("STITCHED")
        return stitched


def stitch_two_frames_togather(stitcher_array):
    temp_stitcher = []
    for i in range(0, len(stitcher_array), 2):
        img1 = stitcher_array[i]
        if (i + 1) > len(stitcher_array) - 1:
            temp_stitcher.append(img1)
            return temp_stitcher
        img2 = stitcher_array[i + 1]
        # stitch img 1 and img 2.
        stitched_image = stitch_image(img1, img2)
        if stitched_image is None:
            print("Stitching Failed")
            continue
        print("Stitching Successful = " + str(i))
        temp_stitcher.append(stitched_image)
    return temp_stitcher


def divide_array_into_sub_array(key_frame, number_of_sub_array = 10):
    key_frames_arrray = []
    jump = int(len(key_frame) / number_of_sub_array)

    for i in range(0, len(key_frame), jump):
        key_frames_arrray.append(key_frame[i: i + jump])

    return key_frames_arrray


def stitch_image_with(key_frames, images_folder_path, out_image_name):
    print("[INFO] stitching images...")
    # initialize OpenCV's image Stitcher object and then perform the image stitching.
    stitcher = cv2.Stitcher.create(mode=1)
    (status, stitched) = stitcher.stitch(key_frames)

    #  Image stitched status.
    if status == 1:
        print(" ERR_NEED_MORE_IMGS = 1,")
    elif status == 2:
        print("ERR_HOMOGRAPHY_EST_FAIL = 2,")
    elif status == 3:
        print("ERR_CAMERA_PARAMS_ADJUST_FAIL = 3")
    elif status == 0:
        print("STITCHED")

    time3 = time.time()
    # cv2.imwrite(images_folder_path + out_image_name, stitched)

    # convert BGR to RGB so that color shown right.
    return stitched
    stitched = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
    plt.imshow(stitched)
    plt.show()

    print('key frames extraction:', time2 - time1, ' sec')
    print('Stitching: ', time3 - time2, ' sec')


# Main part of Code started.

# Section = 1 : Stitch image form given folder.
# Section = 2 : Stitch video frames of given video file.
# Section = 3 : Stitch image form given folder like merge sort.
selection = 2

# Name of stitched image file
out_image_name = "stitched_image.JPG"
# Time counter.
time1 = time.time()

if selection == 1:
    # Absolute path of folder having all images.
    images_folder_path = "/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/Dataset/UAVs/"
    print("[INFO] key frames extraction of images...")
    # Read all images absolute path.
    images_name = read_all_images_name(images_folder_path)
    # Read all images form path and save them in key_frames array.
    key_frames = read_all_images_form(images_name)

    # Time counter.
    time2 = time.time()

    # Perform Final stitching.
    stitch_image_with(key_frames, images_folder_path, out_image_name)
    # optical_flow = IML_optical_flow()
    # optical_flow.optical_flow_of_panaroma(key_frames)

elif selection == 2:
    # This portion perform stitching combining video frames.

    # Absolute path of video file.
    movie_path = path.dataset_path + "Malaria_Dataset_self/SHIF_images/videos/IMG_4284.MOV"
    # this assigment is use to stitched image.
    images_folder_path = movie_path
    # Extract the required frames.
    key_frames = extract_key_frames_from_movie(movie_path)

    # this code divide total framas into 5 subArray the individuall sitiched these images and then finally stitched
    # all sub array stitched images.

    five_array = divide_array_into_sub_array(key_frames, number_of_sub_array=10)
    # Time counter.
    time2 = time.time()
    # key_frames.reverse()

    final_stitched_array = []
    for i in range(len(five_array)):
        temp_stitched = stitch_image_with(five_array[i], images_folder_path, out_image_name)
        final_stitched_array.append(temp_stitched)
        print("stitching Array number = ", i)
        temp_stitched = cv2.cvtColor(temp_stitched, cv2.COLOR_BGR2RGB)
        plt.imshow(temp_stitched)
        plt.show()
    # Perform Final stitching.
    stitch_image_with(final_stitched_array, images_folder_path, out_image_name)

elif selection == 3:
    # Absolute path of folder having all images.
    images_folder_path = path.dataset_path + "Malaria_Dataset_self/SHIF_images/panaroma_3/"
    print("[INFO] key frames extraction of images...")
    # Read all images absolute path.
    images_name = read_all_images_name(images_folder_path)
    # Read all images form path and save them in key_frames array.
    key_frames = read_all_images_form(images_name)
    # Time counter.
    time2 = time.time()
    stitcher_array = stitch_two_frames_togather(key_frames)

    while True:
        if stitcher_array is None:
            print("FINAL STITCHING FAIL EXIT ")
            break
        if len(stitcher_array) < 2:
            stitcher_array = stitch_two_frames_togather(stitcher_array)
            break
        if len(stitcher_array) < 1:
            break
        else:
            # stitch images form jump of 2.
            stitcher_array = stitch_two_frames_togather(stitcher_array)

    time3 = time.time()
    cv2.imwrite(images_folder_path + out_image_name, stitcher_array[0])
    # convert BGR to RGB so that color shown right.
    stitched = cv2.cvtColor(stitcher_array[0], cv2.COLOR_BGR2RGB)
    plt.imshow(stitched)
    plt.show()

    print('key frames extraction:', time2 - time1, ' sec')
    print('Stitching: ', time3 - time2, ' sec')
