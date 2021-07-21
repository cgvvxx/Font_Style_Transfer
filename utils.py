import imageio
import numpy as np
from io import BytesIO
from PIL import Image


def pad_seq(seq, batch_size):
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq


def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


def normalize_image(img):  # normalize (0,255) -> (-1,1)
    normalized = (img / 127.5) - 1
    return normalized


def read_split_image(img):
    mat = imageio.imread(img).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]
    img_B = mat[:, side:]

    return img_A, img_B


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h = img.shape
    enlarged = np.array(Image.fromarray(img).resize([nw, nh]))
    enlarged[np.where(enlarged > 255)] = 255
    enlarged[np.where(enlarged < 0)] = 0

    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]
