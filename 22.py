import cv2 as cv
import numpy as np
import matplotlib.pylab as plt

cap=cv.VideoCapture(0)

hand_filter=cv.CascadeClassifier("d:/hand_recog/hand.xml")
face_filter=cv.CascadeClassifier("d:/hand_recog/face.xml")
#Making Kernel
kernel=cv.getStructuringElement(cv.MORPH_RECT, (5,5))

#Masking Range in YCrCb
Cr_1=np.array([40,135,73])
Cr_2=np.array([130,170,158])

#Masking Range in BGR

BGR_L=np.array([30,30,30])
BGR_U=np.array([150,150,150])

#Masking Range in HSV

HSV_L=np.array([0,50,80])
HSV_U=np.array([120,255,255])

#Calculating Hamming_Distance
'''
target=cv.imread("d:/palm_set/palm_2.jpg")
target_hsv=cv.cvtColor(target, cv.COLOR_BGR2HSV)
target_hist=cv.calcHist(target_hsv, [0,1], None, [257, 80], [0,257, 0,80])
target_gray=cv.cvtColor(target, cv.COLOR_BGR2GRAY)
target_resize=cv.resize(target_gray, (25,25))
target_avg=target_resize.mean()
target_bi=1*(target_resize>target_avg)

def hamming_distance(a,b):#Hamming_distance
      a=a.reshape(1,-1)
      b=b.reshape(1,-1)

      distance=(a!=b).sum()
      return distance

'''

while True:
      #480 * 640
      ref,frame=cap.read()
      frame=cv.GaussianBlur(frame, (3,3),0)
      frame_clone=frame.copy()
      zeros_3=np.zeros_like(frame)
      zeros_4=np.zeros_like(frame)
      
      #Deleting Background
      
      mask_bgr=cv.inRange(frame_clone, BGR_L, BGR_U)
      ref, mask_bgr=cv.threshold(mask_bgr, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

      frame_ycc=cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)
      pre_mask_ycc=cv.inRange(frame_ycc, Cr_1, Cr_2)
      ref, mask_ycc=cv.threshold(pre_mask_ycc, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
      
      #morphorology
      mask_ycc=cv.morphologyEx(mask_ycc, cv.MORPH_CLOSE, kernel, 15)
      mask_ycc=cv.morphologyEx(mask_ycc, cv.MORPH_OPEN, kernel, 15)
      mask_bgr=cv.morphologyEx(mask_bgr, cv.MORPH_CLOSE, kernel, 15)
      mask_bgr=cv.morphologyEx(mask_bgr, cv.MORPH_OPEN, kernel, 15)
      #Bits calculating
      frame_3=cv.bitwise_and(frame, frame, mask=mask_ycc)
      frame_3_gray=cv.cvtColor(frame_3, cv.COLOR_BGR2GRAY)
      aaa=cv.bitwise_and(frame,frame,mask=mask_bgr)
      aaa=cv.bitwise_and(aaa,aaa,mask=mask_ycc)
      cv.imshow("BGR+YCC", aaa)
      
      canny_ycc=cv.Canny(mask_ycc, 100, 200)
      canny_contours, canny_hierarchy=cv.findContours(mask_ycc, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
      canny_list=[]

      #Finding Outter Line of Contours
      for i in range(len(canny_hierarchy[0])):
            if(canny_hierarchy[0][i][2]+canny_hierarchy[0][i][3]<-1):
                  canny_list.append(i)                  

      #for i in range(len(canny_hierarchy[0])) :
      for i in canny_list:
            
            c=canny_contours[i]
            
            mmt=cv.moments(c)
            
            hull=cv.convexHull(c)
            hull_for_defect=cv.convexHull(c, returnPoints=False)
            #Calculating Area, Length, Center
            hull_mmt=cv.moments(hull)
            #Convex Area
            hull_a=hull_mmt['m00']
            a=mmt['m00']
            #Convex Length
            hull_l=cv.arcLength(hull, True)
            l=cv.arcLength(canny_contours[i], True)
            hull_E=0.0
            E=0.0
            
            #Calculating CenterPoints
            
            hull_cx=int(hull_mmt['m10']/hull_mmt['m00'])
            hull_cy=int(hull_mmt['m01']/hull_mmt['m00'])
            
            try:
                  #Calculating Critival Boundary Energy
                  hull_E=(hull_l * hull_l)/ (hull_a)
                  E=(l * l)/a
            except Exception as e:
                  hull_E=0.0
                  E=0.0

            mmask=np.zeros((482, 642), np.uint8)
            
            #Convex Critival Range For Final masking [10 25]
            if hull_E>12 and hull_E<30 and E>15 and E<90 and hull_a>5000 and hull_a<30000:
                  
                  cv.drawContours(zeros_3, [hull], -1, (0,255,255), 3)
                  cv.drawContours(frame_3, [hull], -1, (0,0,255), 3)
                  cv.drawContours(zeros_3, [canny_contours[i]], -1, (0,0,255), 3)
                  cv.drawContours(frame_3,[canny_contours[i]], -1, (0,255,255), 3)
                  cv.putText(zeros_3, "hull_E : %f"%hull_E, (hull_cx, hull_cy), cv.FONT_HERSHEY_PLAIN,2,(255,0,255),3)
                  cv.putText(zeros_3, "E : %f"%E, (hull_cx, hull_cy-30), cv.FONT_HERSHEY_PLAIN, 2,(255,255,0),3)
                  cv.putText(zeros_3, "hull_A : %f"%hull_a, (hull_cx, hull_cy+30), cv.FONT_HERSHEY_PLAIN, 2, (0,255,255),3)
                  cv.circle(zeros_3, (hull_cx,hull_cy), 3,(255,255,255),3)
                  cv.drawContours(zeros_4, [canny_contours[i]], -1, (255,255,255), 3)

                  retval=cv.floodFill(zeros_4, mmask, (hull_cx, hull_cy), (255,255,255), (10,10,10), (10,10,10))
                                      
                  defects=cv.convexityDefects(c,hull_for_defect)
                  for i in range(defects.shape[0]):
                        startP, endP, farthestP, distance=defects[i,0]
                        farthest=tuple(c[farthestP][0])
                        dist=distance/256.0
                        if dist>1:
                              cv.circle(zeros_3, farthest, 5, (255,255,255), 3)
            
            else:
                  cv.putText(zeros_3, "hull_E : %f"%hull_E, (hull_cx, hull_cy), cv.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
                  cv.putText(zeros_3, "hull_A : %f"%hull_a, (hull_cx, hull_cy+30), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255),2)
                  cv.putText(zeros_3, "E : %f"%E, (hull_cx, hull_cy-30), cv.FONT_HERSHEY_PLAIN, 1, (255,255,255),2)
            
      #End of loop

      #Final Masking NEED FIX
      #*******************************************************
      
      #retval=cv.floodFill(zeros_4, mmask, (hull_cx, hull_cy), (255,255,255), (10,10,10), (10,10,10))
      #zeros_4=cv.bitwise_not(zeros_4)\
      cv.imshow("ZEROS_4", zeros_4)
      zeros_4=cv.cvtColor(zeros_4, cv.COLOR_BGR2GRAY)
      fframe=cv.bitwise_and(frame_3, frame_3, mask=zeros_4)

      #**********************************************************

      '''
      #haar cascade
      try:
            hand=hand_filter.detectMultiScale(fframe, 1.3, 5)
            x,y,w,h=hand[0]
            cv.rectangle(frame_clone, (x,y), (x+w, y+h), (0,0,255),3)
      except Exception as e:
            print(str(e))
      '''

      #Termination
      key=cv.waitKey(30) & 0xff                                       
      if key==27:
            break
      #SHOWING
      cv.imshow("FRAME_3", frame_3)
      cv.imshow("Final", fframe)
      '''
      #These 3 are for Testing
      cv.imshow("Zeros_4", zeros_4)
      cv.imshow("Frame_3", frame_3)
      cv.imshow("YCC", mask_ycc)
      '''
      cv.imshow("CONVEX_CONTOUR", zeros_3)
      
      
#end of Image processing
      
cap.release()
cv.destroyAllWindows()