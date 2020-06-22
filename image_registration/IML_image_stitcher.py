import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import imutils
import time
import os
from custom_classes import path, cv_iml

cv2.ocl.setUseOpenCL(False)


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


def remove_black_region(result):
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

    # Finds contours from the binary image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # get the maximum contour area
    c = max(cnts, key=cv2.contourArea)

    # get a bbox from the contour area
    (x, y, w, h) = cv2.boundingRect(c)

    # crop the image to the bbox coordinates
    crop = result[y:y + h, x:x + w]

    # return the cropped image
    return crop


def read_all_images_name(folder_path):
    """

    :param folder_path: this is the path of folder in which we pick all images.
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


def read_all_images_form(images_names):
    # Reading all images form file.
    all_images_array = []
    for image_name in images_names:
        print(image_name)
        img = cv2.imread(image_name)
        img = image_sharpning(img)
        # img = remove_black_region(img)
        # cv2.imwrite(image_name, img)
        all_images_array.append(img)
    return all_images_array


def detectAndDescribe(image, method=None):
    """
    Compute key points and feature descriptors using an specific method
    """

    assert method is not None, "You need to define a feature detection method. Values are: 'sift', 'surf'"

    # detect and extract features from the image
    if method == 'sift':
        descriptor = cv2.xfeatures2d.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.xfeatures2d.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    # get keypoints and descriptors
    (kps, features) = descriptor.detectAndCompute(image, None)

    return kps, features


def createMatcher(method, crossCheck):
    "Create and return a Matcher Object"

    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def matchKeyPointsBF(featuresA, featuresB, method):
    bf = createMatcher(method, crossCheck=True)

    # Match descriptors.
    best_matches = bf.match(featuresA, featuresB)

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    print("Raw matches (Brute force):", len(rawMatches))
    return rawMatches


def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
    bf = createMatcher(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(featuresA, featuresB, 2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m, n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches


def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
    # convert the keypoints to numpy arrays
    kpsA = np.float32([kp.pt for kp in kpsA])
    kpsB = np.float32([kp.pt for kp in kpsB])

    if len(matches) > 4:

        # construct the two sets of points
        ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

        # estimate the homography between the sets of points
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                         reprojThresh)

        return matches, H, status
    else:
        return None


def stitch_two_images(queryImg, trainImg, feature_extractor):
    # Convert Images to gray scale.
    trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)
    queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)
    # Extract features for both images.
    print("[INFO] Extracting Feature ...")
    kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
    kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)
    # Find matching between features.
    featuresA = featuresA[:int(len(featuresA) * 0.3)]
    featuresB = featuresB[:int(len(featuresB) * 0.3)]
    print("[INFO] Find matching between features ...")
    if feature_matching == 'bf':
        matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
    elif feature_matching == 'knn':
        matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
    # Find homography matrix.
    print("[INFO] Find Homography ...")
    M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)

    if M is None:
        print("Error! In Homograpy Matrix.")
    (matches, H, status) = M

    # Apply panorama correction
    width = trainImg.shape[1] + queryImg.shape[1]
    height = trainImg.shape[0] + queryImg.shape[0]

    result = cv2.warpPerspective(trainImg, H, (width, height))
    result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
    crop_image = remove_black_region(result)
    return crop_image


feature_extractor = 'surf'  # one of 'sift', 'surf', 'brisk', 'orb'
feature_matching = 'bf'

images_folder_path = "/Users/qaziammar/Downloads/EXTRACTS/4/"

time1 = time.time()
print("[INFO] Reading Frames ...")
# images_name = path.read_all_files_name_from(images_folder_path, '.jpg')
images_name = read_all_images_name(images_folder_path)
images_array = read_all_images_form(images_name)
stitched_image = images_array[0]

if len(images_array) == 1:
    print("Perform no Stitching")
elif len(images_array) == 2:
    print("stitch only 2 images.")
    queryImg = images_array[0]
    trainImg = images_array[1]
    stitched_image = stitch_two_images(queryImg, trainImg, feature_extractor)
elif len(images_array) > 2:

    queryImg = images_array[0]
    trainImg = images_array[1]
    stitched_image = stitch_two_images(queryImg, trainImg, feature_extractor)
    stitched_image = remove_black_region(stitched_image)
    for i in range(2, len(images_array)):
        print("[INFO] Reading Frames #" + str(i))
        queryImg = stitched_image
        trainImg = images_array[i]
        stitched_image = stitch_two_images(queryImg, trainImg, feature_extractor)
        stitched_image = remove_black_region(stitched_image)

cv2.imwrite(images_folder_path + "out.jpg", stitched_image)
plt.imshow(stitched_image)
plt.show()
