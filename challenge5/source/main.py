

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
import numpy as np
import os
#######################################
config = tf.compat.v1.ConfigProto()   #
config.gpu_options.allow_growth = True#
tf.compat.v1.Session(config=config)   #
#######################################
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

from os import listdir
from os.path import isfile, join


import tensorflow_datasets as tfds

import pandas as pd
import tensorflow.keras as keras
import time
import tensorboard
import matplotlib.pyplot as plt
from tensorflow import keras
import subprocess

currentDir = os.getcwd().replace("source", "")
dataset_path = currentDir + "dataset/train_images/train_images/"

imagesTrainPath = "../dataset/train_images/train_images/"
labelsPath = "../dataset/train_images/train_label.txt"
TRAINING_FILENAMES = [f for f in listdir(imagesTrainPath) if isfile(join(imagesTrainPath, f))]

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_labels(labelsPath, classes):
    df = pd.read_csv(labelsPath)
    lst = df.values.tolist()
    labels = [classes.index(i[0]) for i in lst]
    return labels

def to_oneHot(labels):
    a = labels.astype(int)
    b = np.zeros((a.size, a.max()+1))
    b[np.arange(a.size),a] = 1
    return b

def get_class(modelOutput): #takes the max probability index of a row and turns it into a one hot vector
    b = np.zeros(modelOutput.shape)
    b[np.arange(b.shape[0]), np.argmax(a, axis=1)] = 1
    return b

# print('Tensorflow version ' + tf.__version__)
# print(TRAINING_FILENAMES[0:10])

# os.popen(cp data'.txt' /label)
imgsize = (32,32)
batchsize = 500


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset/train_images/train_images",
    labels = "inferred",
    label_mode = 'categorical',
    validation_split=0.2,
    subset="training",
    shuffle=True,
    seed=123,
    image_size = imgsize,
    batch_size=batchsize)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset/train_images/train_images",
    labels = "inferred",
    label_mode = 'categorical',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=imgsize,
    batch_size=batchsize)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# LR_START = 0.00001
# LR_MAX = 0.00005
# LR_MIN = 0.00001
# LR_RAMPUP_EPOCHS = 5
# LR_SUSTAIN_EPOCHS = 0
# LR_EXP_DECAY = .8

EPOCHS = 15
LR_START = 0.00001
LR_MAX = 0.00009
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 8
LR_SUSTAIN_EPOCHS = 4
LR_EXP_DECAY = .9

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
rng = [i for i in range(25 if EPOCHS<25 else EPOCHS)]
y = [lrfn(x) for x in rng]
# plt.plot(y)
# plt.show()


class convNeuralNet:
    def __init__(self):
        self.model = self.modelInit()

    def modelInit(self):
        rnet = DenseNet121(
            input_shape=(32,32,3),
            weights='imagenet',
            include_top=False
        )
        rnet.trainable = True
        model = keras.models.Sequential()
        
        model.add(keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
        model.add(keras.layers.experimental.preprocessing.RandomRotation(0.20))

        model.add(rnet)
        model.add(keras.layers.GlobalAveragePooling2D())
        
        model.add(keras.layers.Dense(10, activation = 'softmax'))
        
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def modelLoad(self, path):
        self.model = keras.models.load_model(path)
    
    def train(self, dataset, valset, nEpochs):
        self.model.fit(dataset, validation_data=val_ds, epochs = nEpochs, callbacks=[lr_callback],verbose=2)
    
    def predict(self, data_x):
        return self.model.predict(data_x)

    def ajustDropout(self, new_rates):
        if(len(new_rates)==3):
            self.model.layers[2].rate = new_rates[0]
            self.model.layers[5].rate = new_rates[1]
            self.model.layers[11].rate = new_rates[2]
    def ajustRotation(self, newrate):
        self.model.layers[1].factor = newrate
    


modelConv = convNeuralNet()
modelConv.train(train_ds,val_ds, 30)
