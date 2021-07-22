import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import tensorflow as tf
import os
import cv2
import imageio
from PIL import Image
import re

from load_data import TrainDataProvider, embedding_lookup
from model import Encoder, Decoder, Discriminator
from utils import normalize_image


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


def set_imgs_shape(pngs):
      x_shape, y_shape = [], []
      for png in pngs:
        x_shape.append(png.shape[0])
        y_shape.append(png.shape[1])
      x_min = min(x_shape)
      y_min = min(y_shape)
      shaped_pngs = [cv2.resize(png, (y_min, x_min)) for png in pngs]

      return shaped_pngs


def png2gif(save_dir, save_name, pngs, fps=20):
    shaped_pngs = set_imgs_shape(pngs)
    denormed_pngs = [denormalize_img(png) for png in shaped_pngs]
    dir = os.path.join(save_dir, save_name)
    imageio.mimsave(dir, denormed_pngs, fps=fps)


def ckpt_list(ckpt_dir):
    files = glob.glob(os.path.join(ckpt_dir, '*'))
    ckpt_li = [file[:file.rfind('.')] for file in files if file.endswith(".index")]

    return ckpt_li


def char2pngs(ckpt, data_dir, base_dir, embedding_name, font_id_1, font_id_2, grid, char_id):
    pngs = []
    for i in np.arange(0, 1, grid):
        r, f = fixed_img(ckpt, data_dir, base_dir, embedding_name, font_id_1=font_id_1, font_id_2=font_id_2, grid=i,
                         char_id=char_id)
        pngs.append(np.float32(f[0, :, :, 0]))

    return pngs


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


def several_dict(data_dir, font_dict_name):

    font_dict = pd.read_pickle(os.path.join(data_dir, font_dict_name))
    str_dict = dict(zip([i[idx][:-8] for idx, i in enumerate(font_dict)], range(len(font_dict))))
    num_dict = dict(zip(range(len(font_dict)), [i[idx][:-8] for idx, i in enumerate(font_dict)]))

    return font_dict, str_dict, num_dict


def load_str_img(char_id, base_dir, base_name='NotoSans-Regular/NotoSans-Regular_00'):
    basic_font = base_name + hex(ord(str(char_id)))[2:].upper() + '.png'
    img_url = os.path.join(base_dir, basic_font)

    img = imageio.imread(img_url).astype(np.float32)
    img = normalize_image(img)  # (0,255) -> (-1,1)
    img = img.reshape(128, 128, 1)
    img = tf.convert_to_tensor(img, dtype='float32')
    img = tf.reshape(img, [1, 128, 128, 1])

    return img


def find_font_id(tar):
    if type(tar) == int:
        return [tar]
    elif type(tar) == str:
        return [str_dict[tar]]
    else:
        print('type Error!')
        return None


def fixed_img(ckpt, data_dir, base_dir, embedding_name, font_id_1=0, font_id_2=None, grid=0.5, char_id='a'):
    embeddings = pd.read_pickle(os.path.join(data_dir, embedding_name))
    real_src = load_str_img(char_id, base_dir)
    enc_src, enc_lyr = ckpt.encoder.call(real_src, training=False)
    if font_id_2:
        loc_emb_1 = embedding_lookup(embeddings, find_font_id(font_id_1))
        loc_emb_2 = embedding_lookup(embeddings, find_font_id(font_id_2))
        loc_emb = loc_emb_1 * (1 - grid) + loc_emb_2 * grid
    else:
        loc_emb = embedding_lookup(embeddings, find_font_id(font_id_1))
    embd = tf.concat([enc_src, loc_emb], 3)
    fake_tar = ckpt.decoder.call(embd, enc_lyr, training=False)

    return real_src, fake_tar


def word2imgs(word, ckpt, data_dir, base_dir, embedding_name, font_id_1=0, font_id_2=None, grid=0.5):
    imgs = []
    lower_checks = []
    for char in word:
        if char in ['g', 'j', 'p', 'q', 'y']:
            lower_checks.append(True)
        else:
            lower_checks.append(False)
        real_src, fake_tar = fixed_img(ckpt, data_dir, base_dir, embedding_name, font_id_1=font_id_1, font_id_2=font_id_2, grid=grid,
                                       char_id=char)
        imgs.append(fake_tar)

    return imgs, lower_checks


def crop_img(img):
    img_size = img.shape[0]
    full_white = img_size
    col_sum = np.where(full_white - np.sum(img, axis=0) > 1)
    row_sum = np.where(full_white - np.sum(img, axis=1) > 1)

    if np.any((np.diff(col_sum) < np.mean(np.diff(col_sum)))[0]):
        cleaning_col_sum = np.where((np.diff(col_sum) < np.mean(np.diff(col_sum)))[0])[0]
        x1, x2 = col_sum[0][cleaning_col_sum[0]], col_sum[0][cleaning_col_sum[-1] + 1]
    else:
        col_sum = np.where(full_white - np.sum(img, axis=0) > 1)
        x1, x2 = col_sum[0][0], col_sum[0][-1]

    if np.any((np.diff(row_sum) < np.mean(np.diff(row_sum)))[0]):
        cleaning_row_sum = np.where((np.diff(row_sum) < np.mean(np.diff(row_sum)))[0])[0]
        y1, y2 = row_sum[0][cleaning_row_sum[0]], row_sum[0][cleaning_row_sum[-1] + 1]
    else:
        row_sum = np.where(full_white - np.sum(img, axis=1) > 1)
        y1, y2 = row_sum[0][0], row_sum[0][-1]

    cropped_img = img[y1:y2, x1:x2]

    return cropped_img


def cropped_imgs(word_imgs):
    cropped_imgs = []
    for img in word_imgs:
        cropped_imgs.append(crop_img(img[0, :, :, 0]))

    return cropped_imgs


def concat_imgs(word_imgs, lower_check, space_size=5, pad_size=20):
    max_height = max([img.shape[0] for img in word_imgs])
    padded_imgs = []

    if any(lower_check):
        del_max_height = max_height // 5
        max_height += del_max_height

    for idx, (img, check) in enumerate(zip(word_imgs, lower_check)):
        if len(img) == 0:
            continue
        try:
            pad_value = np.max(img)
        except:
            pad_value = 1
        height, width = img.shape
        pad_y_height = max_height - height
        if any(lower_check):
            if check:
                pad_y = np.full((pad_y_height, width), pad_value, dtype=np.float32)
                padded_img = np.concatenate((pad_y, img), axis=0)
            else:
                pad_y = np.full((pad_y_height - del_max_height, width), pad_value, dtype=np.float32)
                padded_img = np.concatenate((pad_y, img), axis=0)
                pad_y = np.full((del_max_height, width), pad_value, dtype=np.float32)
                padded_img = np.concatenate((padded_img, pad_y), axis=0)
        else:
            pad_y = np.full((pad_y_height, width), pad_value, dtype=np.float32)
            padded_img = np.concatenate((pad_y, img), axis=0)

        pad_x = np.full((max_height, space_size), pad_value, dtype=np.float32)
        padded_img = np.concatenate((padded_img, pad_x), axis=1)

        padded_imgs.append(padded_img)

    padded_img = np.concatenate(padded_imgs, axis=1)
    pad_x = np.full((max_height, pad_size), pad_value, dtype=np.float32)
    padded_img = np.concatenate((pad_x, padded_img), axis=1)
    pad_x = np.full((max_height, pad_size - space_size), pad_value, dtype=np.float32)
    padded_img = np.concatenate((padded_img, pad_x), axis=1)
    pad_y = np.full((pad_size, padded_img.shape[1]), pad_value, dtype=np.float32)
    padded_img = np.concatenate((pad_y, padded_img), axis=0)
    padded_img = np.concatenate((padded_img, pad_y), axis=0)

    return padded_img