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


def calcParams(input):
    cnt, _ = cv2.findContours(input, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, (l, _), _ = cv2.minAreaRect(cnt[0])
    t_arr = np.sum(input, axis=1)/255
    t_std = np.std(t_arr[np.nonzero(t_arr)])
    t_mean = np.mean(t_arr[np.nonzero(t_arr)])
    return (l, t_std, t_mean)


img_list = [94, 183, 264, 212, 237, 95, 240]
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
    res_cnt = img_bgr.copy()
    if len(cnts) != 0:
        sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        max_cnt = sorted_cnts[0]
        max_hull = cv2.convexHull(max_cnt)
        cv2.drawContours(res_cnt, [max_cnt], -1, [255, 0, 0], thickness=1)
        cv2.drawContours(res_cnt, [max_hull], -1, [0, 255, 0], thickness=1)

        cv2.drawContours(mask_cnt, [max_cnt], -1, 255, thickness=cv2.FILLED)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 50))
        mask_cnt2 = cv2.morphologyEx(mask_cnt, cv2.MORPH_CLOSE, kernel=kernel, borderValue=0)
        mask_cnt3 = cv2.morphologyEx(mask_cnt, cv2.MORPH_DILATE, kernel=kernel, borderValue=0)

    # Improve ROI Mask
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
    res_cnt = img_bgr.copy()
    if len(cnts) != 0:
        sorted_cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        max_cnt = sorted_cnts[0]
        max_hull = cv2.convexHull(max_cnt)
        cv2.drawContours(mask_cnt, [max_cnt], -1, 255, thickness=cv2.FILLED)
        cv2.drawContours(res_cnt, [max_cnt], -1, [255, 0, 0], thickness=1)
        cv2.drawContours(res_cnt, [max_hull], -1, [0, 255, 0], thickness=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 50))
        mask_cnt = cv2.morphologyEx(mask_cnt, cv2.MORPH_CLOSE, kernel=kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 20))
        mask_cnt = cv2.morphologyEx(mask_cnt, cv2.MORPH_DILATE, kernel=kernel)
        mask_cnt_fin = imfill(mask_cnt)
        img2write.append(mask_blend(img, mask_cnt_fin))

    # Bounding box and crop
    img_bbox = img_bgr.copy()
    bbox = cv2.boundingRect(mask_cnt_fin)
    mom = cv2.moments(mask_cnt_fin, binaryImage=True)
    cx = int(mom["m10"]/mom["m00"])
    cy = int(mom["m01"]/mom["m00"])
    bbox_small = [cx-int(bbox[2]/4), cy-2*int(bbox[3]/8), 2*int(bbox[2]/4), 6*int(bbox[3]/8)]
    cv2.rectangle(img_bbox, bbox_small, color=[255, 0, 0])
    cv2.rectangle(img_bbox, bbox, color=[0, 0, 255])
    mask_box = np.zeros(img.shape, dtype=np.uint8)
    cv2.rectangle(mask_box, bbox_small, color=255, thickness=cv2.FILLED)
    img2write.append(img_bbox)

    # Find Candidate Glands
    # clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    # res_histeq = clahe.apply(mask_img(img, mask_cnt_fin))
    res_histeq = cv2.equalizeHist(mask_img(img, mask_cnt_fin))
    thr, res_thr_histeq = cv2.threshold(res_histeq, 225, 255, cv2.THRESH_BINARY)
    bsize = 51
    res_athr_histeq = cv2.adaptiveThreshold(res_histeq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=bsize, C=-16)
    res_athr_histeq = res_thr_histeq + res_athr_histeq

    # Eliminate borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bsize, bsize))
    mask_cnt_small = cv2.morphologyEx(mask_cnt_fin, cv2.MORPH_ERODE, kernel=kernel)
    res_athr_histeq_mor = mask_img(res_athr_histeq, mask_cnt_small & mask_box)

    # Clean artifacts
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    res_athr_histeq_mor = cv2.morphologyEx(res_athr_histeq_mor, cv2.MORPH_OPEN, kernel=kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 10))
    res_athr_histeq_mor = cv2.morphologyEx(res_athr_histeq_mor, cv2.MORPH_CLOSE, kernel=kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    res_athr_histeq_mor = cv2.morphologyEx(res_athr_histeq_mor, cv2.MORPH_OPEN, kernel=kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 20))
    res_athr_histeq_mor = cv2.morphologyEx(res_athr_histeq_mor, cv2.MORPH_CLOSE, kernel=kernel)
    res_candidate_glands = res_athr_histeq_mor

    img2write.append(res_histeq)
    img2write.append(res_athr_histeq)
    img2write.append(res_candidate_glands)

    # Blob Analysis
    cnts, _ = cv2.findContours(res_candidate_glands, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res_cnt = img_bgr.copy()
    res_glands = np.zeros(img.shape, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    res_dict = dict(l=[], t_mean=[], t_std=[], tort=[])
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        _, (w, h), angle = cv2.minAreaRect(cnt)
        if w < h:
            angle = 90-angle
        else:
            angle = -angle

        if (area > 200 and (angle > 45 or angle < -45)):
            canvas = np.zeros(img.shape, dtype=np.uint8)
            cv2.drawContours(canvas, [cnt], -1, [255, 0, 0], thickness=cv2.FILLED)
            canvas = cv2.morphologyEx(canvas, cv2.MORPH_DILATE, kernel=kernel)
            l, t_std, t_mean = calcParams(canvas)
            res_dict['l'].append(l)
            res_dict['t_std'].append(t_std)
            res_dict['t_mean'].append(t_mean)
            res_glands = res_glands | canvas

    # Final Image
    img_res = img_bgr.copy()
    img_res[:, :, 1] = mask_blend(img_res[:, :, 1], res_glands)
    cv2.rectangle(img_res, bbox_small, color=[255, 0, 0])
    img2write.append(img_res)

    # Grade Calculation
    total_roi = cv2.countNonZero(mask_box*mask_cnt_fin)
    total_gland = cv2.countNonZero(res_glands)

    # Report
    print("--- SAMPLE IMAGE", str(img_no), "---")
    num_gln = len(res_dict['l'])
    print("- Total gland intensity(%)       : ", str(total_gland/total_roi*100))
    print("- Average gland length(px)       : ", str(sum(res_dict['l'])/num_gln))
    print("- Standard dev of length(px)     : ", str(np.std(res_dict['l'])))
    print("- Average gland thickness(px)    : ", str(sum(res_dict['t_mean'])/num_gln))
    print("- Avg of stdev of thickness(px)  : ", str(sum(res_dict['t_std'])/num_gln))
    print("- Gland tortuosities             : ")
    print("-----------------------\n")

    for n, i in enumerate(img2write):
        cv2.imwrite('./img/out/res_raw'+str(img_no)+'_img'+str(n)+'.png', i)
        img2write = []
