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



