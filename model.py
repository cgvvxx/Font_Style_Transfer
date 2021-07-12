import tensorflow as tf


def downsample(filters, kernel_size=3, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.LeakyReLU(0.2))
    result.add(
        tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    return result


def upsample(filters, kernel_size=5, stride=2, apply_batchnorm=True, apply_dropout=False, p=0.5):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.LeakyReLU(0.2))
    result.add(tf.keras.layers.Conv2DTranspose(filters, kernel_size, strides=stride,
                                               padding='same',
                                               kernel_initializer=initializer,
                                               use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(p))

    return result


def downsample_disc(filters, kernel_size=3, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, kernel_size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU(0.2))

    return result


class Encoder(tf.keras.Model):
    def __init__(self, enc_dim=64):
        super(Encoder, self).__init__()
        self.conv1 = downsample(enc_dim, 5, apply_batchnorm=False)
        self.conv2 = downsample(enc_dim * 2, 5)
        self.conv3 = downsample(enc_dim * 4, 4)
        self.conv4 = downsample(enc_dim * 8)
        self.conv5 = downsample(enc_dim * 8)
        self.conv6 = downsample(enc_dim * 8)
        self.conv7 = downsample(enc_dim * 8)

    def call(self, inputs):
        encode_layers = dict()

        x1 = self.conv1(inputs)
        encode_layers['e1'] = x1
        x2 = self.conv2(x1)
        encode_layers['e2'] = x2
        x3 = self.conv3(x2)
        encode_layers['e3'] = x3
        x4 = self.conv4(x3)
        encode_layers['e4'] = x4
        x5 = self.conv5(x4)
        encode_layers['e5'] = x5
        x6 = self.conv6(x5)
        encode_layers['e6'] = x6
        encoded_source = self.conv7(x6)
        encode_layers['e7'] = encoded_source

        return encoded_source, encode_layers
