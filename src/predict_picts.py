import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np
import os

# if len(sys.argv) != 2:
#     print("usage: python predict.py [filename]")
#     sys.exit(1)

# filename = sys.argv[1]
# print('input:', filename)

result_dir = 'results'

img_height, img_width = 150, 150

# load weights
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(150, 150, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


model.load_weights(os.path.join(result_dir, 'smallcnn.h5'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.summary()

# img to 4d-tensor
path = './test/'
files = os.listdir(path)
print(files)
test_list = []
for f in files:
	print(f)
	img = image.load_img(path+f, target_size=(img_height, img_width), grayscale=True)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	
	x = x / 255.0
	pred = model.predict(x)
	print(f,pred)
	# test_list.append(x)
# test_list = np.array(test_list)
# norm
# x = x / 255.0

# print(x)
# print(x.shape)

# pred
# pred = model.predict(test_list)
print(pred)

