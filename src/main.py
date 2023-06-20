import cv2
import glob

path = "./img/"
for ind, f in enumerate(sorted(glob.glob(path+'raw/*.jpeg'))):
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr, _ = cv2.threshold(img, 150, 255, cv2.THRESH_OTSU)
    _, res_thr = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)

    res = cv2.hconcat([img, res_thr])

    cv2.imwrite(path+"out/res"+str(ind)+".png", res)
    # cv2.imshow("name", res)
    # cv2.waitKey(0)
    # break
