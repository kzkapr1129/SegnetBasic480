import tensorflow as tf

def encode_layer(x, filter, add_pooling=True):
    x1 = tf.layers.conv2d(inputs=x, filters=filter, kernel_size=[3, 3], padding="same")
    x1_bn = tf.layers.BatchNormalization()(x1, training=True)
    x1_relu = tf.nn.relu(x1_bn)
    if add_pooling:
        return tf.layers.max_pooling2d(inputs=x1_relu, pool_size=2, strides=2, padding="valid")
    return x1_relu

def decode_layer(x, filter, add_upsampling=True):
    if add_upsampling:
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x1 = tf.layers.conv2d(inputs=x, filters=filter, kernel_size=[3, 3], padding="same")
    x1_bn = tf.layers.BatchNormalization()(x1, training=True)
    return tf.nn.relu(x1_bn)

def out_layer(x, input_w, input_h, classes):
    x1 = tf.layers.conv2d(inputs=x, filters=classes, kernel_size=[1, 1], padding="valid")
    x2 = tf.reshape(x1, [-1, input_w * input_h, classes])
    return tf.nn.softmax(x2, name="output")

def segnet_basic(X, classes):
    x1 = encode_layer(X, 64)
    x2 = encode_layer(x1, 128)
    x3 = encode_layer(x2, 256)
    x4 = encode_layer(x3, 512, False)
    x5 = decode_layer(x4, 512, False)
    x6 = decode_layer(x5, 256)
    x7 = decode_layer(x6, 128)
    x8 = decode_layer(x7, 64)
    x = out_layer(x8, X.get_shape()[1].__int__(), X.get_shape()[2].__int__(), classes)
    return x