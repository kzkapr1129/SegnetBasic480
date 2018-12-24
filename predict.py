"""
画像一枚の認識を実施する
"""

import numpy as np
import keras
from PIL import Image
import dataset
import cv2

from dataset import IMAGE_HEIGHT as H
from dataset import IMAGE_WIDTH as W

CLASSES = 2

def write_image(image, filename):
    """
    認識結果の出力を行う
    """

    floor = [0, 0, 0]
    raspi_box = [0, 255, 0]
    red = image.copy()
    green = image.copy()
    blue = image.copy()
    label_colours = np.array([floor, raspi_box])
    for class_type in range(0, 2):
        red[image == class_type] = label_colours[class_type, 0]
        green[image == class_type] = label_colours[class_type, 1]
        blue[image == class_type] = label_colours[class_type, 2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:, :, 0] = red/1.0
    rgb[:, :, 1] = green/1.0
    rgb[:, :, 2] = blue/1.0
    result_image = Image.fromarray(np.uint8(rgb))
    result_image.save(filename)

def predict(test):
    """
    画像一枚の認識を実施する
    """

    model = keras.models.load_model('seg.h5')
    probs = model.predict(test, batch_size=1)
    prob = probs[0].reshape((H, W, CLASSES)).argmax(axis=2)
    return prob

def main():
    """
    画像一枚の認識を実施する
    """

    data = []
    dataset_obj = dataset.Dataset()
    data.append(dataset_obj.normalized(cv2.imread("test.png")))
    data = np.array(data)
    data = dataset_obj.preprocess_input(data)

    prob = predict(data)
    write_image(prob, "val.png")

if __name__ == '__main__':
    main()
