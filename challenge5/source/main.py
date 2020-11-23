

from datetime import datetime
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
import tensorflow_datasets as tfds

from os import listdir
from os.path import isfile, join

import csv

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
#################################################
# Import the necessary libraries 
from PIL import Image 
from numpy import asarray 


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

def one_hot_to_lables(onehotData):
    return np.argmax(onehotData, axis = 1)

def write_to_file(data, outputFile = "output.csv"):
    myFile = open(outputFile, 'w', newline='')
    with myFile:
        writer = csv.writer(myFile)
        data = [["id", "classes"]] + data
        writer.writerows(data)
    print("Writing to ", outputFile," completed")

# print('Tensorflow version ' + tf.__version__)
# print(TRAINING_FILENAMES[0:10])

# os.popen(cp data'.txt' /label)
imgsize = (32,32)
batchsize = 500

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset/train_images/train_images",
    labels = "inferred",
    label_mode = 'categorical',
    validation_split=0.10,
    subset="training",
    shuffle=True,
    seed=321,
    image_size = imgsize,
    batch_size=batchsize)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "../dataset/train_images/train_images",
    labels = "inferred",
    label_mode = 'categorical',
    validation_split=0.10,
    subset="validation",
    seed=321,
    image_size=imgsize,
    batch_size=batchsize)


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

EPOCHS = 15
# LR_START = 0.00001
# LR_MAX = 0.00005
# LR_MIN = 0.00001
# LR_RAMPUP_EPOCHS = 5
# LR_SUSTAIN_EPOCHS = 0
# LR_EXP_DECAY = .8

LR_START = 0.00001      # bfore 0.0001    
LR_MAX = 0.00045 # bfore 0.0005
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 8
LR_SUSTAIN_EPOCHS = 2 # before 2
LR_EXP_DECAY = .8


# LR_START = 0.00001
# LR_MAX = 0.00015
# LR_MIN = 0.00001
# LR_RAMPUP_EPOCHS = 8
# LR_SUSTAIN_EPOCHS = 2
# LR_EXP_DECAY = .9

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
# def lrfn(epoch):
#     if epoch < LR_RAMPUP_EPOCHS:
#         lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
#     elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
#         lr = LR_MAX
#     elif epoch > (LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS)/2:
#          lr = LR_MAX/2
#     else:
#         lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
#     return lr

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)
rng = [i for i in range(25 if EPOCHS<25 else EPOCHS)]
y = [lrfn(x) for x in rng]
# plt.plot(y)
# plt.show()
import random
def random_invert_img(x, p=0.5):
  if  random.random() < p:
    x = (255-x)
  else:
    x
  return x

def random_invert(factor=0.5):
  return tf.keras.layers.Lambda(lambda x: random_invert_img(x, factor))

# def random_greyscale_convert_img(x, p=0.5):
#   if  random.random() < p:
#     x = tf.image.rgb_to_grayscale(x)
#   else:
#     x
#   return x

# def random_greyscale(factor = 0.5):
#     return tf.keras.layers.Lambda(lambda x: random_greyscale_convert_img(x, factor ))

class convNeuralNet:
    def __init__(self):
        self.model = self.modelInit()

    def modelInit(self):
        rnet = DenseNet121(
            input_shape=(32,32,3),
            weights="imagenet",
            include_top=False,
            pooling=max
        )
        rnet.trainable = True
        model = keras.models.Sequential()
        
        model.add(random_invert(factor=0.1))
        model.add(tf.keras.layers.GaussianNoise(0.12)) #before 0.1
        model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))
        model.add(keras.layers.experimental.preprocessing.RandomFlip("horizontal")) #84.5 horizontal
        model.add(tf.keras.layers.experimental.preprocessing.RandomContrast(0.01))
        
        model.add(rnet)
        model.add(keras.layers.GlobalMaxPooling2D())
        
        model.add(keras.layers.Dense(10, activation = 'sigmoid'))
        
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def modelLoad(self, path):
        self.model = keras.models.load_model(path)
    
    def train(self, dataset, valset, nEpochs):
        self.model.fit(dataset, validation_data=val_ds, epochs = nEpochs, callbacks=[lr_callback, tensorboard_callback],verbose=2)
    
    def predict(self, data_x):
        return self.model.predict(data_x)

    def ajustDropout(self, new_rates):
        if(len(new_rates)==3):
            self.model.layers[2].rate = new_rates[0]
            self.model.layers[5].rate = new_rates[1]
            self.model.layers[11].rate = new_rates[2]
    def ajustRotation(self, newrate):
        self.model.layers[1].factor = newrate
    
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

modelConv = convNeuralNet()
modelConv.train(train_ds,val_ds, 20)

testDir = "../dataset/test_images/content/data/test_imagess/"

test_list = []
for i in range(10000):
    img = Image.open(testDir +"Image_"+str(i+1) + ".png") 
    array = asarray(img)
    test_list.append(array)
    
test_list = np.array(test_list)

prediction = modelConv.predict(test_list)

i = np.arange(prediction.shape[0])
labels = one_hot_to_lables(prediction)
output = np.stack((i, labels), axis = 1)
print(len(labels))
write_to_file(output.tolist(), outputFile = "prediction.csv")
