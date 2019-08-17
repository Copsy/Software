#This is learning_Model
import cv2 as cv
import numpy as np
import os
from PIL import Image
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models

#To read_data
lookup = dict()
reverselookup = dict()
count = 0

data_row=640
data_col=240

#Will be changed 320 x 120

#10_down
for j in os.listdir('d:/leapgestrecog/leapGestRecog/00'):
    if not j.startswith('.'): # If running this code locally, this is to 
                              # ensure you aren't reading in hidden folders
        lookup[j] = count#--->Making_Value 10_down : 9
                            # 01_palm : 0
        reverselookup[count] = j # 0 : 01_palm
                                # 9 : 10_down
        count = count + 1


x_data=[]
y_data=[]#Test img size 640 240
datacount=0#---> Will be 200000
for i in range(0, 10): # Loop over the ten top-level folders
    for j in os.listdir('d:/leapgestrecog/leapGestRecog/0' + str(i) + '/'):#Making_PATH
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('d:/leapgestrecog/leapGestRecog/0' + #d:/leapgestrecog/leapGestRecog0i/0x_????/
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                img = Image.open('d:/leapgestrecog/leapGestRecog/0' + 
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Read in and convert to greyscale
                img = img.resize((320, 240))
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1)

''' 
Checking wherther it is loaded Normally

for i in range(0, 10):
    plt.imshow(x_data[i*200 , :, :])
    plt.title(reverselookup[y_data[i*200 ,0]])
    plt.show()
'''


y_data = to_categorical(y_data)
# x_data : datacount row x cols x depth
x_data = x_data.reshape((datacount, 240, 320, 1))#--->size (20000,120,320,1)
x_data /= 255
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
#split-> For testing split 0.2 : 20%
#x_train : (16000, 120, 320, 1) / x_test : (200,120,320, 1)
#x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)

'''
#Learning_Model Conv->Pooling->conv->pool->flat->Dense->Train
model=models.Sequential(name="Model")
#Layer_1 conv2d->Pooling->ReLU
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(240, 320,1))) 
model.add(layers.MaxPooling2D((2, 2)))
#Layer_2 conv2d->Pooling->ReLU
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())#To make 1D array
model.add(layers.Dense(256, activation='relu'))#Fully_Connected_Layer
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=15, batch_size=100, verbose=1, validation_data=(x_validate, y_validate))
#-----------------------Learning_Layer----------------


[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
print("Accuracy:" + str(acc))
'''