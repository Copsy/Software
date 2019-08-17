import cv2 as cv
import numpy as np
from keras.models import load_model
import time
#import Learning_Model_Test--->#Learning Data Do no import until complete

#Learning_Model
model = load_model('d:/project_kojak-master/models/VGG_cross_validated.h5')

#Masking Range in YCrCb
Cr_1=np.array([40,135,73])
Cr_2=np.array([130,170,158])
kernel=cv.getStructuringElement(cv.MORPH_RECT, (5,5))
bgSubThreshold=50
isBgCaptured=0
x_1,y_1,x_2,y_2=50,50,350,350#300 x 300 x 3 Setting_ROI

prediction = ''
score = 0

gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

def remove_frame(r_frame):
    fgmask = bgModel.apply(r_frame, learningRate=0)
    r_kernel = np.ones((3, 3), np.uint8)
    fgmask = cv.erode(fgmask, r_kernel, iterations=1)
    res = cv.bitwise_and(r_frame, r_frame, mask=fgmask)
    return res

cap=cv.VideoCapture(0)
cap.set(10,200)

def predict_rgb_image(img):
    result = gesture_names[model.predict_classes(img)[0]]
    print(result)
    return (result)


def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score


while True:
    ref, origin_frame=cap.read()
    frame=origin_frame.copy()
    frame=cv.flip(frame,1)
    frame=cv.GaussianBlur(frame, (3,3),0)
    #Frame 480 640 3
    #Pre-Processing->Blur, Masking, Morphology

    frame_ycc=cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
    
    pre_mask_ycc=cv.inRange(frame_ycc,Cr_1,Cr_2)
    ref, mask_ycc=cv.threshold(pre_mask_ycc, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    
    frame=cv.morphologyEx(frame, cv.MORPH_CLOSE, kernel, 15)
    frame=cv.morphologyEx(frame, cv.MORPH_OPEN, kernel, 15)
    
    frame=cv.bitwise_and(frame,frame,mask=mask_ycc)
    
    #reshaped_frame=frame.reshape((1,640,480,3))
    
    if(isBgCaptured==1):
        vv=remove_frame(frame)
        vv=cv.cvtColor(vv,cv.COLOR_BGR2GRAY)
        ref, thresh_mask=cv.threshold(vv,60,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        frame=cv.bitwise_and(frame,frame,mask=thresh_mask)
        roi_thresh=(thresh_mask.copy())[y_1:y_2,x_1:x_2]
        cv.imshow("ROI_thresh", roi_thresh)
    #Pre-Processing Is Done    
    
    cv.rectangle(frame, (x_1,y_1),(x_2,y_2),(255,255,255),3)
    
    ROI=(frame.copy())[x_1:x_2,y_1:y_2]
    #---> size : (240,640)
    cv.imshow("Figure_1",frame)
    cv.imshow("ROI", ROI)#200 x 200 x 3
    #esc==Exit
    key=cv.waitKey(30)
    if key==27:
        break;
    elif key==ord('b'):
        print("B")
        bgModel = cv.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        time.sleep(2)
        isBgCaptured = 1
        print('Background captured')
    elif key==32:#SpaceBar--->Predict In ROI
        print("Space_Bar")
        target=np.stack((roi_thresh,)*3, axis=-1)
        target=cv.resize(target,(224,224))
        target=target.reshape(1,224,224,3)# Data # : 1, 224x224x3
        prediction, score=predict_rgb_image_vgg(target)
    elif key==ord('r'):#Reset
        isBgCaptured=0
        cv.destroyWindow("ROI_thresh")

cv.destroyAllWindows()
cap.release()