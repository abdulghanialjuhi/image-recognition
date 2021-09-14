import numpy as np
import cv2
import os
# import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tensorflow.python.framework import ops


LR = 1e-3
IMAGE_SIZE = 50
MODEL_NAME = 'dogsvscats-{}-{}-model'.format(LR, '6conv-basic-video')
IMG_DIR = '/Users/abdulghanialjuhi/Desktop/image-recognition/image-app/image'

# if os.path.exists('{}.meta'.format(MODEL_NAME)):
#     print('model loaded')

ops.reset_default_graph()

convnet = input_data(shape=[None, IMAGE_SIZE, IMAGE_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)


convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)


convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                     loss='categorical_crossentropy', name='targets')


model = tflearn.DNN(convnet, tensorboard_dir='log')

model.load(MODEL_NAME)

# test_data = process_test_data()

# test_data = np.load('test_data.npy', allow_pickle=True)


def define_img():
    try:
        img = os.listdir(IMG_DIR)[0]
        path = os.path.join(IMG_DIR, img)
        img = cv2.resize(cv2.imread(
        path, cv2.IMREAD_GRAYSCALE), (IMAGE_SIZE, IMAGE_SIZE))
    except cv2.error:
        return 'sorry, something went wrong'
    except Exception:
        return 'Sorry, we can\'t recognize this image'

    data = img.reshape(IMAGE_SIZE, IMAGE_SIZE, 1)
    model_out = model.predict([data])[0]
    if np.argmax(model_out) == 1:
        str_label = 'Dog'
    else:
        str_label = 'Cat'

    return str_label
