import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf

cap=cv.VideoCapture(0)

X=tf.placeholder(tf.uint8, shape=[None,10])

kernel=cv.getStructuringElement(cv.MORPH_RECT, (5,5))

#Masking Range in YCrCb
Cr_1=np.array([40,135,73])
Cr_2=np.array([130,170,158])

while True:
    ref, origin_img=cap.read()
    frame=cv.GaussianBlur(origin_img,(3,3),0)
    frame_ycc=cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    pre_mask_ycc=cv.inRange(frame_ycc, Cr_1, Cr_2)
    ref, mask_ycc=cv.threshold(pre_mask_ycc, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
      
    #morphorology
    mask_ycc=cv.morphologyEx(mask_ycc, cv.MORPH_CLOSE, kernel, 15)
    mask_ycc=cv.morphologyEx(mask_ycc, cv.MORPH_OPEN, kernel, 15)

    #Bits calculating
    frame_3=cv.bitwise_and(frame, frame, mask=mask_ycc)#Our DATA
    re_frame=cv.resize(frame_3,dsize=(240,320),interpolation=cv.INTER_AREA)
    
    key=cv.waitKey(30) & 0xff                                       
    if key==27:
        break
        
    cv.imshow("CONVEX_CONTOUR",frame_3)

      
cap.release()
cv.destroyAllWindows()