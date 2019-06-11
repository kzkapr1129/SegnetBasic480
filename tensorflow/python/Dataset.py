import numpy as np
import csv
import glob
import cv2

IMAGE_HEIGHT = 480
IMAGE_WIDTH = 480
TRAIN_X_BASE_DIR = 'data_img/'
TRAIN_Y_BASE_DIR = 'data_label/'
CLASSES = 2

def loadCSV(filename):
    csv_file = open(filename, "r", encoding="ms932", errors="", newline="")
    csv_file_value = csv.reader(csv_file , delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    values = [v for v in csv_file_value]
    x = [v[0] for v in values]
    y = [v[1] for v in values]
    return np.array(x), np.array(y)

def preprocessX(filepath):
    img = cv2.imread(filepath)
    resizedImg = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
    return resizedImg.astype(np.float) / 128.0 - 1.0 # [0, 255] -> [-1.0, 1.0)

def preprocessY(filepath):
    img = cv2.imread(filepath)
    resizedImg = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))

    one_hot_img = np.zeros([resizedImg.shape[0], resizedImg.shape[1], CLASSES])
    for i in range(IMAGE_HEIGHT):
            for j in range(IMAGE_WIDTH):
                one_hot_img[i, j, resizedImg[i][j]] = 1.0
    return np.reshape(one_hot_img, (IMAGE_HEIGHT*IMAGE_WIDTH, CLASSES))

def loadImagesX(filenames):
    imgs = []
    for filename in filenames:
        img = preprocessX(TRAIN_X_BASE_DIR + filename)
        imgs.append(img)
    return imgs

def loadImagesY(filenames):
    imgs = []
    for filename in filenames:
        img = preprocessY(TRAIN_Y_BASE_DIR + filename)
        imgs.append(img)
    return imgs

def generator(x_train, y_train, num_batch):
	N = x_train.shape[0]
	while True:
		ridxes = np.random.permutation(N)
		for i in range(N-num_batch+1):
			idxes = ridxes[i:i+num_batch if i+num_batch < N else N]
			yield np.array(loadImagesX(x_train[idxes])), np.array(loadImagesY(y_train[idxes]))

def test_generator(x_test, y_test, num_batch):
    N = x_train.shape[0]
	while True:
		ridxes = np.random.permutation(N)
		for i in range(0, N-num_batch+1, num_batch):
			idxes = ridxes[i:i+num_batch if i+num_batch < N else N]
			yield np.array(loadImagesX(x_train[idxes])), np.array(loadImagesY(y_train[idxes]))

def loadTrain():
    return loadCSV("data.csv")