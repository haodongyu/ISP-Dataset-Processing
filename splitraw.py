import imageio
import cv2
import numpy as np
import collections
import os
import PIL.Image
import scipy.io
import rawpy 
import time 
import math
 
MIN_MATCH_COUNT = 4


def splitraw(redmiraw, huaweirgb, cropped_file, compare_matched_file):
    ## (1) prepare data
    huaweirgb_ext = os.path.splitext(huaweirgb)[1]
    redmiraw_ext = os.path.splitext(redmiraw)[1]

    huaweirgb_img = cv2.imread(huaweirgb) # BGR
    #huaweirgb_img = imageio.imread(huaweirgb)
    #print('huaweirgb_img: ', np.shape(huaweirgb_img))
  
    huaweirgb_gray = cv2.cvtColor(huaweirgb_img, cv2.COLOR_RGB2GRAY)
    #print('huaweirgb_gray: ', np.shape(huaweirgb_gray))

    redmiraw_raw = rawpy.imread(redmiraw)
    redmiraw_img = redmiraw_raw.raw_image 
    #print('@redmiraw_img: ', np.shape(redmiraw_img))
 
    # flip raw : up/down
    if 'redmi8' in redmiraw:
        redmiraw_img = np.rot90(redmiraw_img)
        redmiraw_img = np.rot90(redmiraw_img)
    #print('@redmiraw_img: ', np.shape(redmiraw_img))

    # directly to int8, not ok
    #redmiraw_gray = redmiraw_img.astype(np.uint8) 
    redmiraw_gray = cv2.cvtColor(redmiraw_raw.postprocess(), cv2.COLOR_RGB2GRAY)
    #print('redmiraw_gray: ', np.shape(redmiraw_gray))
  
    # raw to float
    redmiraw_img = redmiraw_img.astype(np.float32)
    
    ## (2) Create SIFT object
    sift = cv2.xfeatures2d.SIFT_create()

    ## (3) Create flann matcher
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})

    ## (4) Detect keypoints and compute keypointer descriptors
    kpts1, descs1 = sift.detectAndCompute(redmiraw_gray, None)
    kpts2, descs2 = sift.detectAndCompute(huaweirgb_gray, None)

    ## (5) knnMatch to get Top2
    matches = matcher.knnMatch(descs1, descs2, 2)
    # Sort by their distance.
    matches = sorted(matches, key = lambda x:x[0].distance)

    ## (6) Ratio test, to get good matches.
    good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

    # get redmiraw shape: HxW
    redmiraw_shape = np.shape(redmiraw_gray)
    #print('redmiraw_shape: ', redmiraw_shape) # redmiraw_shape
    H = redmiraw_shape[1]
    W = redmiraw_shape[0]


    canvas = huaweirgb_gray.copy() 
    ## (7) find homography matrix 
    if len(good) > MIN_MATCH_COUNT: 
        src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)  # huawei-rgb
        dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)  # redmi-raw

        # print('src_pts@huaweirgb: {}\n{}'.format(np.shape(src_pts), src_pts))
        # print('dst_pts@redmiraw: {}\n{}'.format(np.shape(dst_pts), dst_pts))

        ## find homography matrix in cv2.RANSAC using good match points
        # https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga4abc2ece9fab9398f2e560d53c8c9780
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

        # print('M: {}\n{}'.format(np.shape(M), M))
        #print('mask: {}\n{}'.format(np.shape(mask), mask))

        ## 掩模，用作绘制计算单应性矩阵时用到的点对
        #matchesMask2 = mask.ravel().tolist()

        ## 计算图1的畸变，也就是在图2中的对应的位置
 
        # Define a rectangle of the redmiraw data with shape [4, 2]
        pts = np.float32([[0, 0], [0, H-1], [W-1, H-1], [W-1, 0]])
        # print('@pts: {}\n{}'.format(np.shape(pts), pts))

        # Reshape rectangle points to [4, 1, 2]
        pts = pts.reshape(-1,1,2)
        # print('@pts-reshaped: {}\n{}'.format(np.shape(pts), pts))
 
        dst = cv2.perspectiveTransform(pts, M)

        ## 绘制边框
        cv2.polylines(canvas, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))

    ## (8) drawMatches
    matched = cv2.drawMatches(redmiraw_gray, kpts1, canvas, kpts2, good, None) #,**draw_params)
    match_original = compare_matched_file.replace('.jpg','_siftmatchingpoints.jpg')
    cv2.imwrite(match_original, matched)

    ## (9) Crop the matched region from scene 
    pts = np.float32([[0, 0], [0, H-1], [W-1, H-1], [W-1, 0]]).reshape(-1,1,2)
    # print('@pts-reshaped: {}\n{}'.format(np.shape(pts), pts))

    dst = cv2.perspectiveTransform(pts, M)
    perspectiveM = cv2.getPerspectiveTransform(np.float32(dst), pts) 
    huaweirgb_cropped = cv2.warpPerspective(huaweirgb_gray, perspectiveM, (H, W)) #borderMode=cv2.BORDER_REPLICATE 
    cv2.imwrite(cropped_file, huaweirgb_cropped)
 
    huaweirgb_cropped_rgb = cv2.warpPerspective(huaweirgb_img, perspectiveM, (H, W)) #borderMode=cv2.BORDER_REPLICATE 
    cv2.imwrite(cropped_file.replace('.jpg', '_rgb.jpg'), huaweirgb_cropped_rgb)
 
    # compare cropped 
    # print('huaweirgb_cropped: ', np.shape(huaweirgb_cropped))
    # print('redmiraw_gray: ', np.shape(redmiraw_gray))

    # cv2.imshow('redmiraw_gray', redmiraw_gray)
    # cv2.waitKey(0)

    compare_matched = np.vstack((redmiraw_gray, huaweirgb_cropped))
    cv2.imwrite(compare_matched_file, compare_matched)

    return redmiraw_img, huaweirgb_cropped_rgb, redmiraw_gray, huaweirgb_cropped
