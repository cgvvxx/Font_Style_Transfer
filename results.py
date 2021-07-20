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
