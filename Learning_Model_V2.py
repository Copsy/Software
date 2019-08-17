import os
import numpy as np
from PIL import Image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
#import matplotlib.pylab as plt
epoch=15
lookup = {}
reverselookup = {}
count = 0
#P_PATH is path that has data
P_PATH="d:/dataset/"

data_row=200
data_col=200
drop_rate=0.5

x_data=[]
y_data=[]
datacount=0
ran=0 # The number of Labels

for j in os.listdir(P_PATH):#A to Z Folder
    lookup[j]=count
    reverselookup[count]=j
    count+=1

ran=count

    
for i in range(0,ran):
    count=0
    tmp_y_value=[]
    for j in os.listdir(P_PATH+str(reverselookup[i])):
        img=Image.open(P_PATH+str(reverselookup[i])+"/"+str(j)).convert("L")
        arr=np.array(img)
        x_data.append(arr)
        count+=1
        
    tmp_y_value=np.full((count,1),i)
    y_data.append(tmp_y_value)#---> y[i].shape == (3000,1)
    datacount+=count
    
x_data=np.array(x_data, dtype=np.float32)#x_data.shape==87000, 200 ,200
y_data=np.array(y_data) #y_data.shape=29 3000 1
y_data=y_data.reshape((datacount,1)) # y_data.shape==87000 1 Per 3000

#datacount == 87000

'''
#Checking whether it is loaded normally

for i in range(0, 29):
    plt.imshow(x_data[i*3000,:,:])
    plt.title(reverselookup[y_data[i*3000,0]])
    plt.show()
'''

y_data=to_categorical(y_data) #--->One hot y_data.shape == 87000 29
x_data=x_data.reshape((datacount,data_row,data_col,1)) # 87000 x 200 x 200 x 1
x_data/=255

x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
#split-> For testing split 0.2 : 20%
#x_train : (16000, 120, 320, 1) / x_test : (200,120,320, 1)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)


#Learning_Model CNN

#Hidden Layer # : 5
'''

For one layers, each layers needs at least 30 minutes
Drop_rate=0.5

'''
model=models.Sequential(name="Model")
#Layer_1 conv2d->Pooling->ReLU
model.add(layers.Conv2D(32, (3, 3), strides=(1,1),padding="SAME", activation='relu', input_shape=(data_row, data_col,1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
#Layer_2 conv2d->Pooling->ReLU
model.add(layers.Conv2D(64, (3, 3), strides=(1,1),padding="SAME", activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Conv2D(128, (3, 3),strides=(1,1), padding="SAME", activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Conv2D(256,(3,3),strides=(1,1), padding="SAME", activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(drop_rate))
model.add(layers.Flatten())#To make 1D array
model.add(layers.Dense(256, activation='relu'))#Fully_Connected_Layer
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(ran, activation='softmax'))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

'''
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
'''
model.fit(x_train,y_train,epochs=epoch,batch_size=100,verbose=1,validation_data=(x_validate,y_validate))
#Verbose is status bar "1" that means Enable, and "2" is Disable
[loss,acc]=model.evaluate(x_test,y_test,verbose=1)
print("ACC : "+str(acc))