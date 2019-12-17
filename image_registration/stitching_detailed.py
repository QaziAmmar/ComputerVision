"""
Stitching sample (advanced)
===========================

Show how to use Stitcher API from python.
"""

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

def iml_imshow(img):
    """

    :param img: img to display.
    :return: None
    """
    plt.imshow(img)
    plt.show()


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
        expos_comp_type = cv.detail.ExposureCompensator_NO
    elif expos_comp == 'gain':
        expos_comp_type = cv.detail.ExposureCompensator_GAIN
    elif expos_comp == 'gain_blocks':
        expos_comp_type = cv.detail.ExposureCompensator_GAIN_BLOCKS
    elif expos_comp == 'channel':
        expos_comp_type = cv.detail.ExposureCompensator_CHANNELS
    elif expos_comp == 'channel_blocks':
        expos_comp_type = cv.detail.ExposureCompensator_CHANNELS_BLOCKS
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
            timelapse_type = cv.detail.Timelapser_AS_IS
        elif timelapse == "crop":
            timelapse_type = cv.detail.Timelapser_CROP
        else:
            print("Bad timelapse method")
            exit()
    else:
        timelapse = False
    range_width = rangewidth
    if features_type == 'orb':
        finder = cv.ORB.create()
    elif features_type == 'surf':
        finder = cv.xfeatures2d_SURF.create()
    elif features_type == 'sift':
        finder = cv.xfeatures2d_SIFT.create()
    elif features_type == 'brisk':
        finder = cv.BRISK_create()
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
        full_img = cv.imread(name)
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
            img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        # (kps, imgFea) = finder.detectAndCompute(img, None)
        imgFea = cv.detail.computeImageFeatures2(finder, img)
        features.append(imgFea)
        img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
        images.append(img)
    if matcher_type == "affine":
        matcher = cv.detail_AffineBestOf2NearestMatcher(False, try_cuda, match_conf)
    elif range_width == -1:
        matcher = cv.detail.BestOf2NearestMatcher_create(try_cuda, match_conf)
    else:
        matcher = cv.detail.BestOf2NearestRangeMatcher_create(range_width, try_cuda, match_conf)
    p = matcher.apply2(features)
    matcher.collectGarbage()
    if save_graph:
        f = open(save_graph_to, "w")
        f.write(cv.detail.matchesGraphAsString(img_names, p, conf_thresh))
        f.close()
    indices = cv.detail.leaveBiggestComponent(features, p, 0.3)
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
        estimator = cv.detail_AffineBasedEstimator()
    else:
        estimator = cv.detail_HomographyBasedEstimator()
    b, cameras = estimator.apply(features, p, None)
    if not b:
        print("Homography estimation failed.")
        exit()
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    if ba_cost_func == "reproj":
        adjuster = cv.detail_BundleAdjusterReproj()
    elif ba_cost_func == "ray":
        adjuster = cv.detail_BundleAdjusterRay()
    elif ba_cost_func == "affine":
        adjuster = cv.detail_BundleAdjusterAffinePartial()
    elif ba_cost_func == "no":
        adjuster = cv.detail_NoBundleAdjuster()
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
        rmats = cv.detail.waveCorrect(rmats, cv.detail.WAVE_CORRECT_HORIZ)
        for idx, cam in enumerate(cameras):
            cam.R = rmats[idx]
    corners = []
    mask = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(0, num_images):
        um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)

    warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper peut etre nullptr?
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)

        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())
    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)
    if cv.detail.ExposureCompensator_CHANNELS == expos_comp_type:
        compensator = cv.detail_ChannelsCompensator(expos_comp_nr_feeds)
    #    compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    elif cv.detail.ExposureCompensator_CHANNELS_BLOCKS == expos_comp_type:
        compensator = cv.detail_BlocksChannelsCompensator(expos_comp_block_size, expos_comp_block_size,
                                                          expos_comp_nr_feeds)
    #    compensator.setNrGainsFilteringIterations(expos_comp_nr_filtering)
    else:
        compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)
    if seam_find_type == "no":
        seam_finder = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_NO)
    elif seam_find_type == "voronoi":
        seam_finder = cv.detail.SeamFinder_createDefault(cv.detail.SeamFinder_VORONOI_SEAM);
    elif seam_find_type == "gc_color":
        seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR")
    elif seam_find_type == "gc_colorgrad":
        seam_finder = cv.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
    elif seam_find_type == "dp_color":
        seam_finder = cv.detail_DpSeamFinder("COLOR")
    elif seam_find_type == "dp_colorgrad":
        seam_finder = cv.detail_DpSeamFinder("COLOR_GRAD")
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
        full_img = cv.imread(name)
        if not is_compose_scale_set:
            if compose_megapix > 0:
                compose_scale = min(1.0, np.sqrt(compose_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_compose_scale_set = True;
            compose_work_aspect = compose_scale / work_scale;
            warped_image_scale *= compose_work_aspect
            warper = cv.PyRotationWarper(warp_type, warped_image_scale)
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
            img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                            interpolation=cv.INTER_LINEAR_EXACT)
        else:
            img = full_img;
        img_size = (img.shape[1], img.shape[0]);
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        image_warped = []
        dilated_mask = cv.dilate(masks_warped[idx], None)
        seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
        mask_warped = cv.bitwise_and(seam_mask, mask_warped)
        if blender == None and not timelapse:
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int))
            elif blend_type == "feather":
                blender = cv.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)
        elif timelapser == None and timelapse:
            timelapser = cv.detail.Timelapser_createDefault(timelapse_type)
            timelapser.initialize(corners, sizes)
        if timelapse:
            matones = np.ones((image_warped_s.shape[0], image_warped_s.shape[1]), np.uint8)
            timelapser.process(image_warped_s, matones, corners[idx])
            pos_s = img_names[idx].rfind("/");
            if pos_s == -1:
                fixedFileName = "fixed_" + img_names[idx];
            else:
                fixedFileName = img_names[idx][:pos_s + 1] + "fixed_" + img_names[idx][pos_s + 1:]
            cv.imwrite(fixedFileName, timelapser.getDst())
        else:
            blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])
    if not timelapse:
        result = None
        result_mask = None
        result, result_mask = blender.blend(result, result_mask)
        cv.imwrite(result_name, result)
        zoomx = 600.0 / result.shape[1]
        dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
        dst = cv.resize(dst, dsize=None, fx=zoomx, fy=zoomx)
        cv.imshow(result_name, dst)
        cv.waitKey()

    print('Done')
    return dst


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


all_images_name = read_all_images_name(
    '/Users/qaziammar/Documents/Thesis/Model_Result_Dataset/Dataset/Malaria_Dataset_self/SHIF_images/frames/')

stitched = IML_stitcher(all_images_name)

iml_imshow(stitched)
