import cv2
import glob
import numpy as np
from scipy.ndimage import rotate

def mask_img(input, mask):
    return ((mask/255)*input).astype(np.uint8)

def mask_blend(input, mask):
    return cv2.addWeighted(input, 1, mask, 0.25, 0)

def imfill(input):
    output = np.zeros(input.shape, dtype=np.uint8)
    cnts, _ = cv2.findContours(input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(output, cnts, -1, 255, thickness=cv2.FILLED)
    return output

img_list = [94, 183, 264, 212, 237]
# img_list = range(100)
filelist = sorted(glob.glob('./img/raw/*.jpeg'))
img2write = []
for img_no in img_list:
    img_bgr = cv2.imread(filelist[img_no])
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Find ROI Mask
    thr, res_thr_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 10))
    res_mor = cv2.morphologyEx(res_thr_img, cv2.MORPH_OPEN, kernel=kernel)
    res_mor2 = cv2.morphologyEx(res_mor, cv2.MORPH_CLOSE, kernel=kernel)
    cnts, _ = cv2.findContours(res_mor2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_cnt = np.zeros(img.shape, dtype=np.uint8)
    mask_hull = np.zeros(img.shape, dtype=np.uint8)
    res_cnt = img_bgr.copy()
    if len(cnts) != 0:
        sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        max_cnt = sorted_cnts[0]
        max_hull = cv2.convexHull(max_cnt)
        cv2.drawContours(mask_cnt, [max_cnt], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask_hull, [max_hull], -1, 255, thickness=cv2.FILLED)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 50))
        mask_cnt2 = cv2.morphologyEx(mask_cnt, cv2.MORPH_CLOSE, kernel=kernel, borderValue=0)
        mask_cnt3 = cv2.morphologyEx(mask_cnt, cv2.MORPH_DILATE, kernel=kernel, borderValue=0)
        cv2.drawContours(res_cnt, [max_cnt], -1, [255, 0, 0], thickness=1)
        cv2.drawContours(res_cnt, [max_hull], -1, [0, 255, 0], thickness=1)

    res_sbl = cv2.Sobel(img, dx=0, dy=1, ddepth=-1, ksize=3)
    _, res_sbl2 = cv2.threshold(res_sbl*mask_cnt3, 0, 255, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    res_sbl_mor = cv2.morphologyEx(res_sbl2, cv2.MORPH_CLOSE, kernel=kernel)
    res_sbl_mor = cv2.morphologyEx(res_sbl_mor, cv2.MORPH_OPEN, kernel=kernel)

    h, w = res_sbl.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 5))
    left = res_sbl_mor[:, 0:int(w/3)]
    kernel2 = rotate(kernel, 150)
    left = cv2.morphologyEx(left, cv2.MORPH_CLOSE, kernel=kernel2)
    middle = res_sbl_mor[:, int(w/3):int(2*w/3)]
    middle = cv2.morphologyEx(middle, cv2.MORPH_CLOSE, kernel=kernel)
    right = res_sbl_mor[:, int(2*w/3):]
    kernel2 = rotate(kernel, 30)
    right = cv2.morphologyEx(right, cv2.MORPH_CLOSE, kernel=kernel2)
    res_sbl_mor2 = cv2.hconcat([left, middle, right])

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 10))
    left = res_sbl_mor2[:, 0:int(w/3)]
    kernel2 = rotate(kernel, 150)
    left = cv2.morphologyEx(left, cv2.MORPH_OPEN, kernel=kernel2)
    middle = res_sbl_mor2[:, int(w/3):int(2*w/3)]
    middle = cv2.morphologyEx(middle, cv2.MORPH_OPEN, kernel=kernel)
    right = res_sbl_mor2[:, int(2*w/3):]
    kernel2 = rotate(kernel, 30)
    right = cv2.morphologyEx(right, cv2.MORPH_OPEN, kernel=kernel2)
    res_sbl_mor3 = cv2.hconcat([left, middle, right])

    cnts, _ = cv2.findContours(res_sbl_mor3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask_cnt = np.zeros(img.shape, dtype=np.uint8)
    mask_hull = np.zeros(img.shape, dtype=np.uint8)
    res_cnt = img_bgr.copy()
    if len(cnts) != 0:
        sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        max_cnt = sorted_cnts[0]
        max_hull = cv2.convexHull(max_cnt)
        cv2.drawContours(mask_cnt, [max_cnt], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(mask_hull, [max_hull], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(res_cnt, [max_cnt], -1, [255, 0, 0], thickness=1)
        cv2.drawContours(res_cnt, [max_hull], -1, [0, 255, 0], thickness=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 50))
        mask_cnt = cv2.morphologyEx(mask_cnt, cv2.MORPH_CLOSE, kernel=kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))
        mask_cnt = cv2.morphologyEx(mask_cnt, cv2.MORPH_DILATE, kernel=kernel)
        mask_cnt = imfill(mask_cnt)

    # Bounding box and crop
    img_bbox = img_bgr.copy()
    bbox = cv2.boundingRect(mask_cnt)
    mom = cv2.moments(mask_cnt, binaryImage=True)
    cx = int(mom["m10"]/mom["m00"])
    cy = int(mom["m01"]/mom["m00"])
    bbox_small = [cx-int(bbox[2]/4), cy-int(bbox[3]/8), 2*int(bbox[2]/4), 5*int(bbox[3]/8)]
    cv2.rectangle(img_bbox, bbox_small, color=[255, 0, 0])
    cv2.rectangle(img_bbox, bbox, color=[0, 0, 255])
    mask_box = np.zeros(img.shape, dtype=np.uint8)
    cv2.rectangle(mask_box, bbox_small, color=255, thickness=cv2.FILLED)
    img2write.append(img_bbox)

    # Hist Eq
    res_heq = cv2.equalizeHist(mask_img(img, mask_cnt))
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))        
    res_clahe = clahe.apply(mask_img(img, mask_cnt))

    histeq = cv2.hconcat([img, res_heq, res_clahe])

    # Thresholding
    thr_in = 225
    thr, res_thr_img = cv2.threshold(img, thr_in, 255, cv2.THRESH_BINARY)
    thr, res_thr_heq = cv2.threshold(res_heq, thr_in, 255, cv2.THRESH_BINARY)
    thr, res_thr_clahe = cv2.threshold(res_clahe, thr_in, 255, cv2.THRESH_BINARY)

    thresholded = cv2.hconcat([res_thr_img, res_thr_heq, res_thr_clahe])

    # Adaptive Threshold
    bsize = 51
    cval = -16
    res_athr_img = cv2.adaptiveThreshold(mask_img(img, mask_cnt), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize = bsize, C = cval)
    res_athr_heq = cv2.adaptiveThreshold(res_heq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize = bsize, C = cval)
    res_athr_clahe = cv2.adaptiveThreshold(res_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize = bsize, C = cval)

    res_athr_img    = res_thr_img   + res_athr_img
    res_athr_heq    = res_thr_heq   + res_athr_heq
    res_athr_clahe  = res_thr_clahe +  res_athr_clahe

    athresholded = cv2.hconcat([res_athr_img, res_athr_heq, res_athr_clahe])
    
    # Eliminate borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bsize, bsize))
    mask_cnt_small = cv2.morphologyEx(mask_cnt, cv2.MORPH_ERODE, kernel=kernel)
    res_athr_img    = mask_img(res_athr_img   ,mask_cnt_small&mask_box)
    res_athr_heq    = mask_img(res_athr_heq   ,mask_cnt_small&mask_box)
    res_athr_clahe  = mask_img(res_athr_clahe ,mask_cnt_small&mask_box)

    # Clean artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    res_athr_img    = cv2.morphologyEx(res_athr_img, cv2.MORPH_OPEN, kernel=kernel)
    res_athr_heq    = cv2.morphologyEx(res_athr_heq, cv2.MORPH_OPEN, kernel=kernel)
    res_athr_clahe  = cv2.morphologyEx(res_athr_clahe, cv2.MORPH_OPEN, kernel=kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 10))
    res_athr_img    = cv2.morphologyEx(res_athr_img, cv2.MORPH_CLOSE, kernel=kernel)
    res_athr_heq    = cv2.morphologyEx(res_athr_heq, cv2.MORPH_CLOSE, kernel=kernel)
    res_athr_clahe  = cv2.morphologyEx(res_athr_clahe, cv2.MORPH_CLOSE, kernel=kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    res_athr_img    = cv2.morphologyEx(res_athr_img, cv2.MORPH_OPEN, kernel=kernel)
    res_athr_heq    = cv2.morphologyEx(res_athr_heq, cv2.MORPH_OPEN, kernel=kernel)
    res_athr_clahe  = cv2.morphologyEx(res_athr_clahe, cv2.MORPH_OPEN, kernel=kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 20))
    res_athr_img    = cv2.morphologyEx(res_athr_img, cv2.MORPH_CLOSE, kernel=kernel)
    res_athr_heq    = cv2.morphologyEx(res_athr_heq, cv2.MORPH_CLOSE, kernel=kernel)
    res_athr_clahe  = cv2.morphologyEx(res_athr_clahe, cv2.MORPH_CLOSE, kernel=kernel)

    img2write.append(cv2.vconcat([histeq, thresholded, athresholded,
                                 cv2.hconcat([res_athr_img, res_athr_heq, res_athr_clahe])
                                 ]))                                      

    for n, i in enumerate(img2write):
        cv2.imwrite('./img/out/res_raw'+str(img_no)+'_img'+str(n)+'.png', i)
        img2write = []
