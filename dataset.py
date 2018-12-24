"""
Datasetクラス
"""

import os
import sys
import cv2
import numpy as np
from keras.applications import imagenet_utils

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 480

class Dataset:
    """
    Segnet用の訓練データを加工し提供する
    """

    def __init__(self, classes=2, csv='data.csv', img_dir="data_img/", label_dir="data_label/"):
        self.classes = classes
        self.csv_file = csv
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.shape = IMAGE_WIDTH * IMAGE_HEIGHT

    def normalized(self, rgb):
        """
        画像データを正規化する
        """

        norm = np.zeros((rgb.shape[0], rgb.shape[1], 3), np.float32)

        blue = rgb[:, :, 0]
        green = rgb[:, :, 1]
        red = rgb[:, :, 2]

        norm[:, :, 0] = cv2.equalizeHist(blue)
        norm[:, :, 1] = cv2.equalizeHist(green)
        norm[:, :, 2] = cv2.equalizeHist(red)

        return norm

    def one_hot_it(self, labels):
        """
        ラベルをone hotに変換する
        """

        one_hot_label = np.zeros([IMAGE_HEIGHT, IMAGE_WIDTH, 2])
        for i in range(IMAGE_HEIGHT):
            for j in range(IMAGE_WIDTH):
                one_hot_label[i, j, labels[i][j]] = 1
        return one_hot_label

    def load_data(self):
        """
        画像、ラベルを読み出す
        """

        data = []
        label = []
        with open(self.csv_file) as file:
            lines = file.readlines()
            lines = [line.split(',') for line in lines]

        img_dir_full = os.path.join(os.getcwd(), self.img_dir)
        label_dir_full = os.path.join(os.getcwd(), self.label_dir)

        for line in lines:
            img_path = (img_dir_full + line[0]).strip()
            label_path = (label_dir_full + line[1]).strip()
            data.append(self.normalized(cv2.imread(img_path)))
            label.append(self.one_hot_it(cv2.imread(label_path)[:, :, 0]))
            sys.stdout.write(".")

        return np.array(data), np.array(label)

    def preprocess_input(self, image):
        """Preprocesses a tensor or Numpy array encoding a batch of images.
        # Arguments
            x: Input Numpy or symbolic tensor, 3D or 4D.
                The preprocessed data is written over the input data
                if the data types are compatible. To avoid this
                behaviour, `numpy.copy(x)` can be used.
            data_format: Data format of the image tensor/array.
            mode: One of "caffe", "tf" or "torch".
                - caffe: will convert the images from RGB to BGR,
                    then will zero-center each color channel with
                    respect to the ImageNet dataset,
                    without scaling.
                - tf: will scale pixels between -1 and 1,
                    sample-wise.
                - torch: will scale pixels between 0 and 1 and then
                    will normalize each channel with respect to the
                    ImageNet dataset.
        # Returns
            Preprocessed tensor or Numpy array.
        # Raises
            ValueError: In case of unknown `data_format` argument.
        """
        return imagenet_utils.preprocess_input(image)

    def reshape_labels(self, label):
        """
        2次元の画像データを一次元に変形する
        """
        return np.reshape(label, (len(label), self.shape, self.classes))
