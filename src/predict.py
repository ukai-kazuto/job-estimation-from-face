import os
import sys
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing import image
import numpy as np

if len(sys.argv) != 2:
    print("usage: python predict.py [filename]")
    sys.exit(1)

filename = sys.argv[1]
print('input:', filename)

result_dir = 'results'

img_height, img_width = 150, 150

# 学習済みの重みをロード
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
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


model.load_weights(os.path.join(result_dir, 'smallcnn.h5'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.summary()

# 画像を読み込んで4次元テンソルへ変換
img = image.load_img(filename, target_size=(img_height, img_width), grayscale=True)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# 学習時にImageDataGeneratorのrescaleで正規化したので同じ処理が必要！
# これを忘れると結果がおかしくなるので注意
x = x / 255.0

# print(x)
# print(x.shape)

# クラスを予測
# 入力は1枚の画像なので[0]のみ
pred = model.predict(x)[0]
print(pred)