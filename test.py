import cv2
import matplotlib.pyplot as plt
from skimage import feature
import joblib
import matplotlib.image as mimg
import matplotlib.pyplot as plt

svm_model=joblib.load('svm_face_train_modelnew.pkl')
num_of_sample = 8
vid = cv2.VideoCapture(0)
#check,frame1=vid.read()
#cv2.imshow("cap",frame)

#path='/home/student/Desktop/orl_face/u4/8.png'
#im = mimg.imread(path)
# haar cascade for frontal face
face_cascade = cv2.CascadeClassifier('/home/student/Desktop/haarcascade_frontalface_default.xml')
iter1=0
while(iter1<num_of_sample):
    r,frame = vid.read();# capture a single frame
    frame = cv2.resize(frame,(640,480)) # resizig the frame
    im1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)# gray scale conversion of
    # color image
    face=face_cascade.detectMultiScale(im1)
    for x,y,w,h in (face):
        # [255,0,0] #[B,G,R] 0 to 255 
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),[0,0,255],4)
        iter1=iter1+1
        im_f = im1[y:y+h,x:x+w]
        im_f = cv2.resize(im_f,(112,92))#orl face matching size
        
        feat,hog_image = feature.hog(im_f,orientations=8,pixels_per_cell=(16,16),
                                     visualise=True,cells_per_block=(1,1))
        val1=svm_model.predict(feat.reshape(1,-1))
        print(val1)
        str1=" "
        if val1[0]==41:
            str1="sushma"   
            print(str1)
        else:
            str1="others"
        cv2.putText(frame,str1,(x,y),cv2.FONT_ITALIC,1,(255,0,255),2,cv2.LINE_AA)      
    cv2.imshow('frame',frame)
cv2.waitKey()
vid.release() 
cv2.destroyAllWindows()  
        
#plt.subplot(1,1,1)
#plt.imshow(im,cmap='gray')
'''if int(val1[0])==1:
    str1="u1"

elif int(val1[0]==39):
    str1='u39'
else:
    str1="others"
print(str1) '''
