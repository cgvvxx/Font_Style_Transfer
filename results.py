import pandas as pd
import tensorflow as tf
import os
import cv2
import imageio
from PIL import Image

from load_data import TrainDataProvider, embedding_lookup
from model import Encoder, Decoder, Discriminator


def denormalize_img(img):
    img = (img + 1) * 127.5
    out = img.astype('uint8')
    return out


def save_img(img, save_dir, img_name):
    cv2.imwrite(os.path.join(save_dir, img_name), img)


def add_zero(total_num, idx):
    total_num = len(str(total_num))
    num = len(str(idx))
    if num < total_num:
        return '0' * (total_num - num) + str(idx)
    else:
        return str(idx)


def ckpt_load(ckpt_dir, data_dir, embedding_name, ckpts=None, ckpt_idx=None, latest=False):
    g_optimizer = tf.keras.optimizers.Adam()
    d_optimizer = tf.keras.optimizers.Adam()

    embeddings = pd.read_pickle(os.path.join(data_dir, embedding_name))
    fonts_num = len(embeddings)

    En = Encoder()
    De = Decoder()
    Dis = Discriminator(cat_num=fonts_num)

    ckpt = tf.train.Checkpoint(generator_optimizer=g_optimizer, discriminator_optimizer=d_optimizer,
                               encoder=En, decoder=De, discriminator=Dis)

    if latest:
        ckpt.restore(tf.train.latest_checkpoint(ckpt_dir))
    else:
        ckpt.restore(ckpts[ckpt_idx])

    return ckpt


def one_batch_plot(batch, ckpt, embeddings, save_dir=None, save=False, save_name=None):
    real_tar = tf.reshape(batch[2][:, :, :, 1], [1, 128, 128, 1])
    real_src = tf.reshape(batch[2][:, :, :, 0], [1, 128, 128, 1])

    enc_src, enc_lyr = ckpt.encoder.call(real_src, training=False)
    loc_emb = embedding_lookup(embeddings, batch[0])
    embd = tf.concat([enc_src, loc_emb], 3)
    fake_tar = ckpt.decoder.call(embd, enc_lyr, training=False)
    img_list = [real_src, real_tar, fake_tar]
    title = ['Input Image', 'Ground Truth({0})'.format(batch[0][0]), 'Predicted Image']

    if save:
        img = np.concatenate([real_tar.numpy()[0, :, :, 0], fake_tar.numpy()[0, :, :, 0]], axis=1)
        img = denormalize_img(img)
        save_img(img, save_dir, save_name)

    plt.figure(figsize=(10, 8))

    for i in range(len(img_list)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img_list[i][0, :, :, 0], cmap='gray')
        plt.title(title[i])

        frame = plt.gca()
        frame.axes.xaxis.set_ticklabels([])
        frame.axes.yaxis.set_ticklabels([])

    plt.show()


def ret_img(batch, ckpt, embeddings):
    real_tar = tf.reshape(batch[-1][:, :, :, 1], [1, 128, 128, 1])
    real_src = tf.reshape(batch[2][:, :, :, 0], [1, 128, 128, 1])

    enc_src, enc_lyr = ckpt.encoder.call(real_src, training=False)
    loc_emb = embedding_lookup(embeddings, batch[0])
    embd = tf.concat([enc_src, loc_emb], 3)
    fake_tar = ckpt.decoder.call(embd, enc_lyr, training=False)

    concat_image = tf.concat([real_src, fake_tar, real_tar], axis=2)

    return concat_image[0, :, :, 0]


def random_plot_save(ckpt, num, data_dir, embedding_name, train_name, save=False, save_dir=None, shuffle=True):
    batch_size = 1
    embeddings = pd.read_pickle(os.path.join(data_dir, embedding_name))
    data_provider = TrainDataProvider(data_dir, train_name=train_name)
    train_batch_iter = data_provider.get_train_iter(batch_size, shuffle=shuffle, with_charid=True)

    idx = 0
    while True:
        idx += 1

        train_check = next(train_batch_iter)
        save_name = add_zero(num, idx) + '.png'
        one_batch_plot(train_check, ckpt, embeddings, save_dir=save_dir, save=save, save_name=save_name)

        if idx == num:
            break