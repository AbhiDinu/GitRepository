import numpy as np
import cv2

def imageread():
    img=cv2.imread("F:\\Sensovision\\Change_Detection_FinalCode_Latest\\0450.bmp",1)
    cv2.imshow("Abhi_image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

'''
    Applying the Gaussian filter
'''
def gaussian_filter():
    img=cv2.imread("F:\\Sensovision\\Change_Detection_FinalCode_Latest\\0450.bmp",1)
    cv2.imshow("Abhi_image",img)
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.float32)/25
    dst = cv2.filter2D(img2,-1,kernel)
    cv2.imshow("Gray_image",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def video_show():
    #global img3
    #img_record = cv2.VideoCapture("F:\\My_Programmer\\Python\\Opencv_Python\\HDFS_Architecture.mp4")
    #img_record = cv2.VideoCapture("HDFS_Architecture.mp4")
    kernel = np.ones((5,5),np.float32)/25
    img_record = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    if img_record.isOpened():
        ret, img3 = img_record.read()
        print("Video file is opened and Shown in window")
        ret = True
    else:
        print("Video file is not opened")
        ret = False
    while ret:
            img3 = cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(img3, 1.3, 5)
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #for (x,y,w,h) in faces:
            #    cv2.putText(img3,'FACE',(x-w,y-h), font, 0.5, (11,255,255), 2)
            for (x,y,w,h) in faces:
                cv2.rectangle(img3,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = img3[y:y+h, x:x+w]
                eye = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eye:
                    cv2.rectangle(roi_gray,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            #blurr_img = cv2.filter2D(img3,-1,kernel)
            #blurr_img = cv2.blur(img3, (5,5))
            cv2.imshow("Abhishek_video",img3)
            ret, img3 = img_record.read()
            key = cv2.waitKey(10)
            if key == 27: # exit on ESC
                break
    img_record.release()            
    cv2.destroyAllWindows()

       
            
