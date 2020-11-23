

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
dataset_path = currentDir + "dataset/train/train/"

imagesTrainPath = "../dataset/train/train/"
labelsPath = "../dataset/train_images/train_label.txt"

# print(TRAINING_FILENAMES)
#################################################
# Import the necessary libraries 
from PIL import Image 
from numpy import asarray 

CSV_PATH = "../dataset/train_kaggle.csv"

data_csv = pd.read_csv(CSV_PATH)

def getClasses(dataframe):
    datalist = dataframe.values.tolist()
    classes = []
    for i in datalist:
        if i[1] not in classes:
            classes.append(i[1])
    return classes

CLASSES = getClasses(data_csv)

# for i in range(len(CLASSES)):
#     subprocess.call(["mkdir", imagesTrainPath + str(i)])

datalist = data_csv.values.tolist()

# for i in range(len(datalist)):
#     oldImage = imagesTrainPath + str(datalist[i][0]) + ".png"

#     newImage = imagesTrainPath + str(CLASSES.index(datalist[i][1])) +  "/" + str(datalist[i][0]) + ".png"
#     subprocess.call(["cp", oldImage, newImage])

# labels = load_labels(labelsPath=labelsPath, classes = CLASSES)
# print(labels[0])



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
        data = [["Id", "Category"]] + data
        writer.writerows(data)
    print("Writing to ", outputFile," completed")

# print('Tensorflow version ' + tf.__version__)
# print(TRAINING_FILENAMES[0:10])

# os.popen(cp data'.txt' /label)
imgsize = (224,224)
batchsize = 25

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    imagesTrainPath,
    labels = "inferred",
    label_mode = 'categorical',
    validation_split=0.15,
    subset="training",
    shuffle=True,
    seed=123,
    image_size = imgsize,
    batch_size=batchsize)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    imagesTrainPath,
    labels = "inferred",
    label_mode = 'categorical',
    validation_split=0.15,
    subset="validation",
    seed=123,
    image_size=imgsize,
    batch_size=batchsize)

# test_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     "../dataset/train_images/train_images",
#     labels = "inferred",
#     label_mode = 'categorical',
#     validation_split=0.1,
#     subset="validation",
#     seed=123,
#     image_size=imgsize,
#     batch_size=batchsize)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

EPOCHS = 25
# LR_START = 0.00001
# LR_MAX = 0.000010
# LR_MIN = 0.00001
# LR_RAMPUP_EPOCHS = 2
# LR_SUSTAIN_EPOCHS = 0
# LR_EXP_DECAY = .8

LR_START = 0.00001      # bfore 0.0001    
LR_MAX = 0.00035 # bfore 0.0005
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 8
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8


# # LR_START = 0.00001
# # LR_MAX = 0.00015
# # LR_MIN = 0.00001
# # LR_RAMPUP_EPOCHS = 8
# # LR_SUSTAIN_EPOCHS = 2
# # LR_EXP_DECAY = .9

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
# # plt.plot(y)
# # plt.show()


class convNeuralNet:
    def __init__(self, classes):
        self.model = self.modelInit()
        self.classes = classes
    def modelInit(self):
        rnet = DenseNet121(
            input_shape=(224,224,3),
            weights="imagenet",
            include_top=False,
        )
        rnet.trainable = True
        model = keras.models.Sequential()
        
        model.add(tf.keras.layers.GaussianNoise(30))
        model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./255))
        # model.add(keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"))
        model.add(tf.keras.layers.experimental.preprocessing.RandomContrast(0.99))
        model.add(keras.layers.experimental.preprocessing.RandomRotation(0.85))

        model.add(rnet)
        model.add(keras.layers.GlobalAveragePooling2D())
        
        model.add(keras.layers.Dense(len(CLASSES), activation = 'softmax'))
        
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


def number_class_to_string(data):
    return CLASSES[data]

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

modelConv = convNeuralNet(CLASSES)
modelConv.train(train_ds,val_ds, 8)


testDir = "../dataset/val/val/"
# TEST_FILENAMES = [f for f in listdir(testDir) if isfile(join(testDir, f))]

CSV_TEST_PATH = "../dataset/val_sample_submission.csv"
images_test_tag = pd.read_csv(CSV_TEST_PATH).values.tolist()
print()
test_list = []
test_id = []
for i in range(len(images_test_tag)):
    img = Image.open(testDir + str(images_test_tag[i][0]) + '.png')
    array = asarray(img)
    test_list.append(array)
    test_id.append(images_test_tag[i][0])
    

test_list = np.array(test_list)

prediction = modelConv.predict(test_list)
test_id = np.array(test_id)
labels = one_hot_to_lables(prediction)

labels_list = labels.tolist()
lables_list = [CLASSES[i] for i in labels_list]
lables_array = np.array(lables_list)

output = np.stack((test_id, lables_array), axis = 1)
write_to_file(output.tolist(), outputFile = "prediction.csv")