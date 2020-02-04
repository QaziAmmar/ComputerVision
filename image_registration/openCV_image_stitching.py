import matplotlib.pyplot as plt
import cv2
from sys import exit
import time
import numpy as np
import os

from custom_classes.black_region_remove_class import removeBlackRegion
from custom_classes import path
# from optical_flow import IML_optical_flow
# This current working code of images stitching.

path.init()

# only required varibale to run the code is.
# images_folder_path = "path of datafolder"

__author__ = "Qazi Ammar Arshad"
__email__ = "qaziammar.g@gmail.com"
__status__ = "This is working code of python that stitched, images automatically. Link of code: " \
             "https://www.pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/ "


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
            if '.JPG' in file:
                all_images_name.append(r + file)
    # return sorted(all_images_name, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
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
        zero_count = np.sum(temp_img_row < 13)
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
    cropped_image = image[int(row_limit * 1): row - int(row_limit * 1),
                    int(col_limit * 1): col - int(col_limit * 1), :]
    # Return cropped Image..
    return cropped_image


def define_row_col_crop_limit_for_panaroma(img, percentage_limit=6):
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
    row_limit_top = 0
    col_limit_top = 0
    row_limit_lower = 0
    col_limit_lower = 0

    # calculate upper limit cut limit of image
    # Set loop (jump = 10) to speed up the loop.
    for i in range(0, row, 10):
        temp_img_row = gray[i, :]

        # Count the number of pixels that has less value then 10 (consider as black pixel.)
        zero_count = np.sum(temp_img_row < 13)
        # Count the number of non_zero pixel in current image row.
        non_zero_count = col - zero_count

        # Percentage of zero_pixel in a col.
        percentage = (zero_count / col) * 100
        # If percentage reaches our defined threshold then terminate the loop.
        if percentage < percentage_limit:
            row_limit_top = i
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
            col_limit_top = j
            break

        # calculate lower cut limit of image
        # Set loop (jump = 10) to speed up the loop.
    for k in range(row - 1, 0, -10):
        temp_img_row = gray[k, :]

        # Count the number of pixels that has less value then 10 (consider as black pixel.)
        zero_count = np.sum(temp_img_row < 13)
        # Count the number of non_zero pixel in current image row.
        non_zero_count = col - zero_count

        # Percentage of zero_pixel in a col.
        percentage = (zero_count / col) * 100
        # If percentage reaches our defined threshold then terminate the loop.
        if percentage < percentage_limit:
            row_limit_lower = k
            break

    # Set loop (jump = 10) to speed up the loop.
    for l in range(col - 1, 0, -10):
        temp_img_col = gray[:, l]

        # Count the number of pixels that has less value then 10 (consider as black pixel.)
        zero_count = np.sum(temp_img_col < 10)
        # Count the number of non_zero pixel in current image col.
        non_zero_count = col - zero_count

        # Percentage of zero_pixel in a row
        percentage = (zero_count / row) * 100
        # If percentage reaches our defined threshold then terminate the loop.
        if percentage < percentage_limit:
            col_limit_lower = l
            break

    # Return number of rows and cols to cropped.
    return row_limit_top, col_limit_top, row_limit_lower, col_limit_lower


def remove_non_zero_row_col_for_panaroma(image):
    """

    This function take image, first find crop point of row and col of image and then crop image from
    calculated row and col point/limit.
    :param image: input image.
    :return: image that cropped and have no black region.
    """
    # Find cut points of image.
    row_limit_top, col_limit_top, row_limit_lower, col_limit_lower = define_row_col_crop_limit_for_panaroma(image)
    row, col, ch = image.shape
    if row < col:
        print("cut will be assign at row level")
        # Crop image form calculated point of row and col.
        cropped_image = image[int(row_limit_top * 1): row_limit_lower, :, :]
    else:
        cropped_image = image[:, int(col_limit_top * 1): col_limit_lower, :]

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
        img = removeBlackRegion(img)
        # for non panaroma.
        # img = remove_non_zero_row_col_of(img)

        # for panaroam.
        # img = remove_non_zero_row_col_for_panaroma(img)
        # img = image_sharpning(img)
        # cv2.imwrite(image_name, img)
        all_images_array.append(img)
    # exit()
    return all_images_array


def extract_key_frames_from_movie(movie_path):
    # Load video form given path.
    cap = cv2.VideoCapture(movie_path)
    # Count Number of Frames form video.
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # This Array contains only selected frame of video.
    key_frames = []

    # Read video frame by jump of 10.
    for i in range(0, int(frame_count), 10):
        # Lode i frame form video.
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        # print('Position:', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        _, frame = cap.read()

        # Comment/Uncomment the required method to apply.
        # frame = cv2.transpose(frame)
        # frame = image_resize(frame, height=300)
        frame = removeBlackRegion(frame)
        frame = remove_non_zero_row_col_of(frame)

        key_frames.append(frame)
        cv2.imwrite(path.dataset_path + "/Malaria_Dataset_self/p_falcipraum_plus/frames/" + str(i) + ".jpg", frame)
    exit()
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


def divide_array_into_sub_array(key_frame, number_of_sub_array=10):
    key_frames_arrray = []
    jump = int(len(key_frame) / number_of_sub_array)

    for i in range(0, len(key_frame), jump):
        key_frames_arrray.append(key_frame[i: i + jump])

    return key_frames_arrray


def stitch_image_with(key_frames, images_folder_path="", out_image_name=""):
    print("[INFO] stitching images...")
    # initialize OpenCV's image Stitcher object and then perform the image stitching.
    stitcher = cv2.Stitcher.create(mode=1)
    (status, stitched) = stitcher.stitch(key_frames)

    #  Image stitched status.
    if status == 1:
        print("ERR_NEED_MORE_IMGS = 1,")
    elif status == 2:
        print("ERR_HOMOGRAPHY_EST_FAIL = 2,")
    elif status == 3:
        print("ERR_CAMERA_PARAMS_ADJUST_FAIL = 3")
    elif status == 0:
        print("STITCHED")

    if status != 0:
        return
    time3 = time.time()

    # convert BGR to RGB so that color shown right.
    stitched = cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB)
    plt.imshow(stitched)
    plt.show()

    print('key frames extraction:', time2 - time1, ' sec')
    print('Stitching: ', time3 - time2, ' sec')
    cv2.imwrite(images_folder_path + out_image_name, stitched)
    return stitched


def IML_stitcher(img_names, preview=False, try_cuda=False, work_megapix=0.6, features='brisk',
                 matcher='affine', estimator='affine', match_conf=0.3, conf_thresh=0.1,
                 ba='affine', ba_refine_mask='xxxxx', wave_correct='no', save_graph=None,
                 warp='plane', seam_megapix=0.1, seam='no', compose_megapix=-1, expos_comp='no',
                 expos_comp_nr_feeds=1, expos_comp_nr_filtering=2, expos_comp_block_size=32,
                 blend='multiband', blend_strength=5, output='result.jpg', timelapse=None,
                 rangewidth=-1):
    # stitcher->setEstimator(makePtr < detail::AffineBasedEstimator > ()); // estimator
    # stitcher->setWaveCorrection(false);  //wave_correct
    # stitcher->setFeaturesMatcher(makePtr < detail::AffineBestOf2NearestMatcher > (false, false)); //matcher
    # stitcher->setBundleAdjuster(makePtr < detail::BundleAdjusterAffinePartial > ()); //ba
    # stitcher->setWarper(makePtr < AffineWarper > ());
    # stitcher->setExposureCompensator(makePtr < detail::NoExposureCompensator > ()); //expos_comp

    img_names = img_names
    print(img_names)
    preview = preview
    try_cuda = try_cuda
    work_megapix = work_megapix
    seam_megapix = seam_megapix
    compose_megapix = compose_megapix
    conf_thresh = conf_thresh
    features_type = features
    matcher_type = matcher
    estimator_type = estimator
    ba_cost_func = ba
    ba_refine_mask = ba_refine_mask
    wave_correct = wave_correct
    if wave_correct == 'no':
        do_wave_correct = False
    else:
        do_wave_correct = True
    if save_graph is None:
        save_graph = False
    else:
        save_graph = True
        save_graph_to = save_graph
    warp_type = warp
    if expos_comp == 'no':
        expos_comp_type = cv2.detail.ExposureCompensator_NO
    elif expos_comp == 'gain':
        expos_comp_type = cv2.detail.ExposureCompensator_GAIN
    elif expos_comp == 'gain_blocks':
        expos_comp_type = cv2.detail.ExposureCompensator_GAIN_BLOCKS
    elif expos_comp == 'channel':
        expos_comp_type = cv2.detail.ExposureCompensator_CHANNELS
    elif expos_comp == 'channel_blocks':
        expos_comp_type = cv2.detail.ExposureCompensator_CHANNELS_BLOCKS
    else:
        print("Bad exposure compensation method")
        exit()
    expos_comp_nr_feeds = expos_comp_nr_feeds
    expos_comp_nr_filtering = expos_comp_nr_filtering
    expos_comp_block_size = expos_comp_block_size
    match_conf = match_conf
    seam_find_type = seam
    blend_type = blend
    blend_strength = blend_strength
    result_name = output
    if timelapse is not None:
        timelapse = True
        if timelapse == "as_is":
            timelapse_type = cv2.detail.Timelapser_AS_IS
        elif timelapse == "crop":
            timelapse_type = cv2.detail.Timelapser_CROP
        else:
            print("Bad timelapse method")
            exit()
    else:
        timelapse = False
    range_width = rangewidth
    if features_type == 'orb':
        finder = cv2.ORB.create()
    elif features_type == 'surf':
        finder = cv2.xfeatures2d_SURF.create()
    elif features_type == 'sift':
        finder = cv2.xfeatures2d_SIFT.create()
    elif features_type == 'brisk':
        finder = cv2.BRISK_create()
    else:
        print("Unknown descriptor type")
        exit()
    seam_work_aspect = 1
    full_img_sizes = []
    features = []
    images = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False;
    for name in img_names:
        full_img = cv2.imread(name)
        if full_img is None:
            print("Cannot read image ", name)
            exit()
        full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        if work_megapix < 0:
            img = full_img
            work_scale = 1
            is_work_scale_set = True
        else:
            if is_work_scale_set is False:
                work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
                is_work_scale_set = True
            img = cv2.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale,
                             interpolation=cv2.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        # (kps, imgFea) = finder.detectAndCompute(img, None)
        imgFea = cv2.detail.computeImageFeatures2(finder, img)
        features.append(imgFea)
        img = cv2.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_LINEAR_EXACT)
        images.append(img)
    if matcher_type == "affine":
        matcher = cv2.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
    elif range_width == -1:
        matcher = cv2.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
    else:
        matcher = cv2.detail.BestOf2NearestRangeMatcher_create(range_width, try_cuda, match_conf)
    p = matcher.apply2(features)
    matcher.collectGarbage()
    if save_graph:
        f = open(save_graph_to, "w")
        f.write(cv2.detail.matchesGraphAsString(img_names, p, conf_thresh))
        f.close()
    indices = cv2.detail.leaveBiggestComponent(features, p, 0.3)
    img_subset = []
    img_names_subset = []
    full_img_sizes_subset = []
    num_images = len(indices)
    for i in range(len(indices)):
        img_names_subset.append(img_names[indices[i, 0]])
        img_subset.append(images[indices[i, 0]])
        full_img_sizes_subset.append(full_img_sizes[indices[i, 0]])
    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;
    num_images = len(img_names)
    if num_images < 2:
        print("Need more images")
        exit()

    if estimator_type == "affine":
        estimator = cv2.detail_AffineBasedEstimator()
    else:
        estimator = cv2.detail_HomographyBasedEstimator()
    b, cameras = estimator.apply(features, p, None)
    if not b:
        print("Homography estimation failed.")
        exit()
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    if ba_cost_func == "reproj":
        adjuster = cv2.detail_BundleAdjusterReproj()
    elif ba_cost_func == "ray":
        adjuster = cv2.detail_BundleAdjusterRay()
    elif ba_cost_func == "affine":
        adjuster = cv2.detail_BundleAdjusterAffinePartial()
    elif ba_cost_func == "no":
        adjuster = cv2.detail_NoBundleAdjuster()
    else:
        print("Unknown bundle adjustment cost function: ", ba_cost_func)
        exit()
    adjuster.setConfThresh(1)
    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)
    if not b:
        print("Camera parameters adjusting failed.")
        exit()
    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    sorted(focals)
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2
    if do_wave_correct:
        rmats = []
        for cam in cameras:
            rmats.append(np.copy(cam.R))
        rmats = cv2.detail.waveCorrect(rmats, cv2.detail.WAVE_CORRECT_HORIZ)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]
    corners = []
    mask = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(0, num_images):
        um = cv2.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)

    warper = cv2.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper peut etre nullptr?
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)

        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())
    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)
    if cv2.detail.ExposureCompensator_CHANNELS == expos_comp_type:
        compensator = cv2.detail_ChannelsCompensator(expos_comp_nr_feeds)
    #    compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    elif cv2.detail.ExposureCompensator_CHANNELS_BLOCKS == expos_comp_type:
        compensator = cv2.detail_BlocksChannelsCompensator(expos_comp_block_size, expos_comp_block_size,
                                                           expos_comp_nr_feeds)
    #    compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    else:
        compensator = cv2.detail.ExposureCompensator_createDefault(expos_comp_type)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)
    if seam_find_type == "no":
        seam_finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_NO)
    elif seam_find_type == "voronoi":
        seam_finder = cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM);
    elif seam_find_type == "gc_color":
        seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR")
    elif seam_find_type == "gc_colorgrad":
        seam_finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    elif seam_find_type == "dp_color":
        seam_finder = cv2.detail_DpSeamFinder("COLOR")
    elif seam_find_type == "dp_colorgrad":
        seam_finder = cv2.detail_DpSeamFinder("COLOR_GRAD")
    if seam_finder is None:
        print("Can't create the following seam finder ", seam_find_type)
        exit()
    seam_finder.find(images_warped_f, corners, masks_warped)
    imgListe = []
    compose_scale = 1
    corners = []
    sizes = []
    images_warped = []
    images_warped_f = []
    masks = []
    blender = None
    timelapser = None
    compose_work_aspect = 1
    for idx, name in enumerate(
            img_names):  # https://github.com/opencv/opencv/blob/master/samples/cpp/stitching_detailed.cpp#L725 ?
        full_img = cv2.imread(name)
        if not is_compose_scale_set:
            if compose_megapix > 0:
                compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_compose_scale_set = True;
            compose_work_aspect = compose_scale / work_scale;
            warped_image_scale *= compose_work_aspect
            warper = cv2.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(img_names)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (full_img_sizes[i][0] * compose_scale, full_img_sizes[i][1] * compose_scale)
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R);
                corners.append(roi[0:2])
                sizes.append(roi[2:4])
        if abs(compose_scale - 1) > 1e-1:
            img = cv2.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                             interpolation=cv2.INTER_LINEAR_EXACT)
        else:
            img = full_img;
        img_size = (img.shape[1], img.shape[0]);
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv2.INTER_LINEAR, cv2.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv2.INTER_NEAREST, cv2.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        image_warped = []
        dilated_mask = cv2.dilate(masks_warped[idx], None)
        seam_mask = cv2.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv2.INTER_LINEAR_EXACT)
        mask_warped = cv2.bitwise_and(seam_mask, mask_warped)
        if blender == None and not timelapse:
            blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            dst_sz = cv2.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv2.detail.Blender_createDefault(cv2.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv2.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
            elif blend_type == "feather":
                blender = cv2.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)
        elif timelapser == None and timelapse:
            timelapser = cv2.detail.Timelapser_createDefault(timelapse_type)
            timelapser.initialize(corners, sizes)
        if timelapse:
            matones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
            timelapser.process(image_warped_s, matones, corners[idx])
            pos_s = img_names[idx].rfind("/");
            if pos_s == -1:
                fixedFileName = "fixed_" + img_names[idx];
            else:
                fixedFileName = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
            cv2.imwrite(fixedFileName, timelapser.getDst())
        else:
            blender.feed(cv2.UMat(image_warped_s), mask_warped, corners[idx])
    if not timelapse:
        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        cv2.imwrite(result_name, result)
        zoomx = 600.0 / result.shape[1]
        dst = cv2.normalize(src=result, dst=None, alpha=255., norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dst = cv2.resize(dst, dsize=None, fx=zoomx, fy=zoomx)

    print('Done')
    return dst


# Main part of Code started.

# Selection = 1 : Stitch image form given folder.
# Selection = 2 : Stitch video frames of given video file.
# Selection = 3 : Stitch image form given folder like merge sort.
# Selection = 4 : Stitch image form given folder by combining sub stitched images arrays.
selection = 1

# Name of stitched image file
out_image_name = "stitched_image.jpg"
# Time counter.
time1 = time.time()

if selection == 1:
    # Absolute path of folder having all images.
    images_folder_path = "/Users/qaziammar/Documents/test_code/"
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
    movie_path = path.dataset_path + "/Malaria_Dataset_self/p_falciparum/panaromas_i7/"
    # this assigment is use to stitched image.
    images_folder_path = movie_path
    # Extract the required frames.
    key_frames = extract_key_frames_from_movie(movie_path)

    # Perform Final stitching.
    stitch_image_with(key_frames, images_folder_path, out_image_name)

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

elif selection == 4:
    # Absolute path of folder having all images.
    images_folder_path = "/Users/qaziammar/Documents/frames/"
    print("[INFO] key frames extraction of images...")
    # Read all images absolute path.
    images_name = read_all_images_name(images_folder_path)
    key_frames = read_all_images_form(images_name)
    # Time counter.
    time2 = time.time()

    five_array = divide_array_into_sub_array(key_frames, number_of_sub_array=10)
    final_stitched_array = []

    for i in range(len(five_array)):
        print("stitching Array number = ", i)
        temp_stitched = stitch_image_with(five_array[i])
        # temp_stitched = remove_non_zero_row_col_of(temp_stitched)
        final_stitched_array.append(temp_stitched)
        # iml_imshow(temp_stitched)
        cv2.imwrite("/Users/qaziammar/Documents/results/sub_stitched" + str(i) + ".jpg",
                    cv2.cvtColor(temp_stitched, cv2.COLOR_BGR2RGB))
    # Perform Final stitching.
    final_stitched = stitch_image_with(final_stitched_array)
    # iml_imshow(final_stitched)
    print("end of code of test sttich")
