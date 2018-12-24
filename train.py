"""
segnet-basicの訓練を行う
"""

import os
import glob
import numpy as np
import keras
from model import segnet_basic
import dataset
from dataset import IMAGE_WIDTH as W
from dataset import IMAGE_HEIGHT as H
import tensorflow as tf

input_shape = (H, W, 3)
classes = 2
epochs = 2
batch_size = 1
data_shape = input_shape[0] * input_shape[1]

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(session)

def main():
    print("loading data...")
    ds = dataset.Dataset()
    train_X, train_y = ds.load_data()

    train_X = ds.preprocess_input(train_X)
    train_y = ds.reshape_labels(train_y)

    net = segnet_basic()
    net.compile(loss="categorical_crossentropy",
                optimizer="adadelta",
                metrics=["accuracy"])
    net.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, verbose=1)
    net.save('seg.h5')

if __name__ == '__main__':
    main()
