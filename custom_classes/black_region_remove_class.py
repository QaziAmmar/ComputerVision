import cv2
import numpy as np


def removeBlackRegion(img):
    """
       This function remove the black region form image by cropping largest contours form the image.
       :param img: input image
       :return: image with removed black region but not removed black regions completely we need to apply some
       thresholding to rows and col to completely remove the black region.
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


def crop_points_for_black_color(img, percentage_limit=6):
    """
    :param img: Input image.
    :param percentage_limit: allowed percentage of black pixels in one row.
    :return: crop point of image to fully remove black region.
    """
    # Convert image RGB to Grayscale.
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # Get total rows and cols of image.
    row, col = gray.shape

    # Crop point of Image eg. row_limit = 200 (crop top and bottom row/col or image.)
    row_limit_top = 0
    col_limit_top = 0
    row_limit_lower = row - 1
    col_limit_lower = col - 1

    # calculate upper limit cut limit of image
    # Set loop (jump = 10) to speed up the loop.
    for i in range(0, row, 10):
        temp_img_row = gray[i, :]

        # Count the number of pixels that has less value then 13 (consider as black pixel.)
        zero_count = np.sum(temp_img_row < 13)
        # Count the number of non_zero pixel in current image row.
        non_zero_count = row - zero_count

        # Percentage of zero_pixel in a col.
        percentage = (zero_count / len(temp_img_row)) * 100
        # If percentage reaches our defined threshold then terminate the loop.
        if percentage < percentage_limit:
            row_limit_top = i
            break

    # Set loop (jump = 10) to speed up the loop.
    for j in range(0, col, 10):
        temp_img_col = gray[:, j]

        # Count the number of pixels that has less value then 13 (consider as black pixel.)
        zero_count = np.sum(temp_img_col < 13)
        # Count the number of non_zero pixel in current image col.
        non_zero_count = col - zero_count

        # Percentage of zero_pixel in a col
        percentage = (zero_count / len(temp_img_col)) * 100
        # If percentage reaches our defined threshold then terminate the loop.
        if percentage < percentage_limit:
            col_limit_top = j
            break

        # calculate lower cut row_limit_lower
        # Set loop (jump = 10) to speed up the loop.
    for k in range(row - 1, 0, -10):
        temp_img_row = gray[k, :]

        # Count the number of pixels that has less value then 20 (consider as black pixel.)
        zero_count = np.sum(temp_img_row < 70)
        # Count the number of non_zero pixel in current image row.
        non_zero_count = row - zero_count

        # Percentage of zero_pixel in a row.
        percentage = (zero_count / len(temp_img_row)) * 100
        # If percentage reaches our defined threshold then terminate the loop.
        if percentage < 3:
            row_limit_lower = k
            break

    # calculate lower cut col_limit_lower
    # Set loop (jump = 10) to speed up the loop.
    for l in range(col - 1, 0, -10):
        temp_img_col = gray[:, l]

        # Count the number of pixels that has less value then 13 (consider as black pixel.)
        zero_count = np.sum(temp_img_col < 13)
        # Count the number of non_zero pixel in current image col.
        non_zero_count = col - zero_count

        # Percentage of zero_pixel in a row
        percentage = (zero_count / len(temp_img_col)) * 100
        # If percentage reaches our defined threshold then terminate the loop.
        if percentage < percentage_limit:
            col_limit_lower = l
            break

    # Return number of rows and cols to cropped.
    return row_limit_top, col_limit_top, row_limit_lower, col_limit_lower