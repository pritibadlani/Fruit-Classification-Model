# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 18:12:24 2020

@author: HP
"""


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


classifier= Sequential()
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu'))

classifier.add(Dense(units=131, activation='softmax'))


classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('fruits-360/Training',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical') 



test_set = test_datagen.flow_from_directory('fruits-360/Test',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 1,
                         validation_data = test_set,
                         validation_steps = 2000)


 
import numpy as np
from keras.preprocessing import image

import matplotlib.pyplot as plt

test_image=image.load_img('fruits-360/test-multiple_fruits/apple_grape.jpg',target_size = (64, 64),color_mode='rgb')
test_image=image.img_to_array(test_image)
test_image= np.expand_dims(test_image,axis=0)
result= classifier.predict(test_image)
result

training_set.class_indices

test_image1=image.load_img('fruits-360/test-multiple_fruits/apples4.jpg',target_size = (64, 64),color_mode='rgb')
test_image1=image.img_to_array(test_image1)
test_image1= np.expand_dims(test_image1,axis=0)
result1= classifier.predict(test_image1)
result1


test_image2=image.load_img('fruits-360/test-multiple_fruits/Bananas(lady_finger)1.jpg',target_size = (64, 64),color_mode='rgb')
test_image2=image.img_to_array(test_image2)
test_image2= np.expand_dims(test_image2,axis=0)
result2= classifier.predict(test_image2)

test_image3=image.load_img('fruits-360/test-multiple_fruits/cherries_wax3.jpg',target_size = (64, 64),color_mode='rgb')
test_image3=image.img_to_array(test_image3)
test_image3= np.expand_dims(test_image3,axis=0)
result3= classifier.predict(test_image3)

val=[index for index,value in enumerate(result[0]) if value ==1]
from keras.models import load_model

classifier.save('fruits.h5')  

# returns a compiled model
# identical to the previous one
model = load_model('fruits.h5')
