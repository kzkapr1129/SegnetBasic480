"""
Segnet-basicのモデルを返却する
"""

from keras import layers
from keras import models
from dataset import IMAGE_HEIGHT as H
from dataset import IMAGE_WIDTH as W

def segnet_basic(input_shape=(H, W, 3), classes=2):
    """
    Segnet-basicのモデルを返却する
    """

    network = models.Sequential()

    ########## エンコード部

    # 畳み込み1
    network.add(layers.Conv2D(64, (3, 3), padding="same", input_shape=input_shape))
    network.add(layers.BatchNormalization())
    network.add(layers.Activation("relu"))
    network.add(layers.MaxPool2D(pool_size=(2, 2)))

    # 畳み込み2
    network.add(layers.Conv2D(128, (3, 3), padding="same"))
    network.add(layers.BatchNormalization())
    network.add(layers.Activation("relu"))
    network.add(layers.MaxPool2D(pool_size=(2, 2)))

    # 畳み込み3
    network.add(layers.Conv2D(256, (3, 3), padding="same"))
    network.add(layers.BatchNormalization())
    network.add(layers.Activation("relu"))
    network.add(layers.MaxPool2D(pool_size=(2, 2)))

    # 畳み込み4
    network.add(layers.Conv2D(512, (3, 3), padding="same"))
    network.add(layers.BatchNormalization())
    network.add(layers.Activation("relu"))

    ########## デコード部

    # 転置畳み込み1
    network.add(layers.Conv2D(512, (3, 3), padding="same"))
    network.add(layers.BatchNormalization())
    network.add(layers.Activation("relu"))

    # 転置畳み込み2
    network.add(layers.UpSampling2D(size=(2, 2)))
    network.add(layers.Conv2D(256, (3, 3), padding="same"))
    network.add(layers.BatchNormalization())
    network.add(layers.Activation("relu"))

    # 転置畳み込み3
    network.add(layers.UpSampling2D(size=(2, 2)))
    network.add(layers.Conv2D(128, (3, 3), padding="same"))
    network.add(layers.BatchNormalization())
    network.add(layers.Activation("relu"))

    # 転置畳み込み4
    network.add(layers.UpSampling2D(size=(2, 2)))
    network.add(layers.Conv2D(64, (3, 3), padding="same"))
    network.add(layers.BatchNormalization())
    network.add(layers.Activation("relu"))

    # 出力
    network.add(layers.Conv2D(classes, (1, 1), padding="valid"))
    network.add(layers.Reshape((input_shape[0] * input_shape[1], classes)))
    network.add(layers.Activation("softmax"))

    return network
