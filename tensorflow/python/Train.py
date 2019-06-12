import tensorflow as tf
import numpy as np
from Model import segnet_basic, encode_layer, decode_layer
from Dataset import loadTrain, generator
from Dataset import IMAGE_HEIGHT, IMAGE_WIDTH, CLASSES
import os
from glob import glob

def get_latest_modified_file_path(dirname):
    target = os.path.join(dirname, 'model_[0-9]*.ckpt.index')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    latest_modified_file_path = sorted(files, key=lambda files: files[1])[-1]
    return latest_modified_file_path[0][:-6]

def save_checkpoint(index):
    saver = tf.train.Saver()
    checkpoint_name = "checkpoints/model_{}.ckpt".format(index)
    saver.save(sess, checkpoint_name)
    print("saved " + checkpoint_name)

# 教師データ読み込み
x_train, y_train = loadTrain()

# 定数
N = x_train.shape[0]
BASE_IR = 0.001
BATCH_SIZE = 1
EPOCHS = 2
epoch_steps = N / BATCH_SIZE
max_itr = int(epoch_steps * EPOCHS)

# ジェネレータ生成
gen = generator(x_train, y_train, BATCH_SIZE)

# Input
X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], "input")
Y = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT*IMAGE_WIDTH, CLASSES])

# model
y = segnet_basic(X, CLASSES)

# 評価関数
cross_entropy = -tf.reduce_sum(tf.log(y) * Y, axis=1)
cross_entropy_mean = tf.reduce_mean(cross_entropy)
train_step = tf.train.RMSPropOptimizer(BASE_IR).minimize(cross_entropy)
acc = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(Y, y))

with tf.Session() as sess:
    try:
        latest_filename = get_latest_modified_file_path("checkpoints")
        saver = tf.train.Saver()
        saver.restore(sess, latest_filename)
        print("init valiables with " + latest_filename)
    except IndexError:
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        print("init valiables without checkpoint")

    try:
        for i in range(max_itr):
            bx, by = gen.__next__()
            train_loss, train_acc, step = sess.run([cross_entropy_mean, acc, train_step], feed_dict={X: bx, Y: by})
            print("loss={:.5f} acc={:.5f}".format(train_loss, train_acc))

            i2 = (i % int(epoch_steps))
            if i != 0 and i2 == 0:
                save_checkpoint(i / int(epoch_steps))

    except KeyboardInterrupt:
        pass
    
    print("saving ...", end="")
    save_checkpoint(EPOCHS)

    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        ['output']
    )

    with open('frozen_graph.pb', 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

    print("[ok]")

