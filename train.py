import numpy as np
import cv2
import matplotlib.image as mimg
import matplotlib.pyplot as plt
from skimage import feature
from sklearn import svm
import pickle
train_data=np.zeros((7*41,280))
train_label=np.zeros((7*41))
count=-1
#plt.figure(1)
#plt.ion()
# feature extraction 
for i in range(1,42):
    for j in range(1,8):
        plt.cla()
        count=count+1
        path='/home/student/Desktop/ml/orl_face/u%d/%d.png'%(i,j)
       # path = './orl_face/orl_face/u%d/%d.png'%(i,j)
        im = mimg.imread(path)
        feat,hog_image = feature.hog(im,orientations=8,pixels_per_cell=(16,16),visualise=True,cells_per_block=(1,1))
        train_data[count,:]=feat.reshape(1,-1)
        train_label[count]=i
        plt.subplot(2,1,1)
        plt.imshow(im,cmap='gray')
        plt.subplot(2,1,2)
        plt.imshow(hog_image,cmap='gray')
        plt.pause(0.1)
        print(i,j)

# model creation
svm_model=svm.SVC(kernel='poly',gamma=0.001)

# train the model
svm_model=svm_model.fit(train_data,train_label)
f=open('svm_face_train_modelnew.pkl','wb')
pickle.dump(svm_model,f)

print('training done ')
