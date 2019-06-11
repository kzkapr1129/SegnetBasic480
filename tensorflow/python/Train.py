import tensorflow as tf
import numpy as np
from Model import segnet_basic, encode_layer, decode_layer
from Dataset import loadTrain, generator
from Dataset import IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES

# 教師データ読み込み
x_train, y_train = loadTrain()

# 定数
N = x_train.shape[0]
BASE_IR = 0.001
BATCH_SIZE = 3
EPOCHS = 1
max_itr = int(N / BATCH_SIZE * EPOCHS)

# ジェネレータ生成
gen = generator(x_train, y_train, BATCH_SIZE)

# Input
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], "input")
Y = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH, CLASSES])

# model
y = segnet_basic(X, CLASSES)

# 評価関数
loss = tf.losses.softmax_cross_entropy(Y, y)
train_step = tf.train.RMSPropOptimizer(BASE_IR).minimize(loss)
acc = tf.keras.metrics.categorical_accuracy(Y, y)

with tf.Session() as sess:
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    try:
        for i in range(max_itr):
            bx, by = gen.__next__()
            train_loss, step = sess.run([loss, train_step], feed_dict={X: bx, Y: by})
            print("loss={}".format(train_loss))

    except KeyboardInterrupt:
        pass
    
    print("saving ...", end="")
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        ['output']
    )

    with open('frozen_graph.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

    print("[ok]")

