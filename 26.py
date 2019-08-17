import cv2 as cv
import numpy as np
#import Learning_Model_V2 #--->Dont import until complete this program
from Learning_Model_V2 import model
'''
Leaning_Model_Test is CNN (N_Hidden_Layer-->Inside : Conv & Max_Padding)
Using ReLU & Adam Optimizer

'''
#Camera -> 640x480
"""

Setting ROI and that images is interperted change to alphabet by Using CNN
When "SpaceBar" button is pressed, that ROI is shoted

"""
x_1, y_1, x_2, y_2=70,70,270,270#ROI location-->200x200

M_kernel=cv.getStructuringElement(cv.MORPH_RECT, (3,3))

Label=["A","B","C","D","E","F","G","H","I","J","K"]

cap=cv.VideoCapture(0)

while True:
    ref, origin_img=cap.read()
    
    #Rre-Processing Blur->Morphology_CLOSE ->Morphology_OPEN
    img=cv.GaussianBlur(origin_img,(3,3),0)
    img=cv.morphologyEx(img,cv.MORPH_CLOSE,M_kernel,10)
    img=cv.morphologyEx(img,cv.MORPH_OPEN, M_kernel,10)
    
    img=cv.flip(img,1)
    
    cv.rectangle(img,(x_1,y_1),(x_2,y_2),(255,255,255),3)
    
    ROI=(img.copy())[y_1:y_2, x_1:x_2]#Deep Copy
    
    key=cv.waitKey(30)
    
    cv.imshow("Figure_1", img)
    
    if key==27:#ESC
        break;
    elif key==32: #SpaceBar
        print("LOCK")
        test_img=ROI.copy()#Deep_Copy
        test_img=cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)#--->1Channel 200 x 200 x1
        #test_img=cv.resize(test_img ,(200,200))
        cv.imshow("TestImg",test_img)
        test_img=np.array(test_img, dtype=np.float32)
        test_img /= 255
        test_img=test_img.reshape((1,200,200,1))
        
        result=model.predict(test_img,verbose=1)
        index=np.argmax(result)
        print(Label[index])
        print(result)
        print("DONE")
    
cv.destroyAllWindows()    
cap.release()
