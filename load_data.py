import pickle as pickle
import numpy as np
import random
import os
import tensorflow as tf
from utils import pad_seq, bytes_to_file, read_split_image
from utils import shift_and_resize_image, normalize_image


def get_batch_iter(examples, batch_size, augment, with_charid=True):

    padded = pad_seq(examples, batch_size)
    

    def process(img):
      img = bytes_to_file(img)
      try :
        img_A, img_B = read_split_image(img)
        if augment :
        # to be shifted
          w, h = img_A.shape
          multiplier = random.uniform(1.00, 1.20)
          # add an eps to prevent cropping issue
          nw = int(multiplier * w) + 1
          nh = int(multiplier * h) + 1
          shift_x = int(np.ceil(np.random.uniform(0.01, nw-w)))
          shift_y = int(np.ceil(np.random.uniform(0.01, nh-h)))
          img_A = shift_and_resize_image(img_A, shift_x, shift_y, nw, nh) # zittering
          img_B = shift_and_resize_image(img_B, shift_x, shift_y, nw, nh) # zittering
          img_A = normalize_image(img_A)
          img_A = img_A.reshape(len(img_A), len(img_A[0]), 1) 
          img_B = normalize_image(img_B)
          img_B = img_B.reshape(len(img_B), len(img_B[0]), 1) 
    
          return np.concatenate([img_A, img_B], axis=2) 

      finally :
          img.close()


    def batch_iter(with_charid=with_charid):

      for i in range(0, len(padded), batch_size):
        batch = padded[i: i + batch_size]
        labels = [e[0] for e in batch]
        if with_charid:
          charid = [e[1] for e in batch]
          image = [process(e[2]) for e in batch]
          image = np.array(image).astype(np.float32)
          image = tf.convert_to_tensor(image, dtype='float32')

          yield [labels, charid, image] # [labels, image]를 쌓아서 return 함
        else:
          image = [process(e[1]) for e in batch]
          image = np.array(image).astype(np.float32)
          image = tf.convert_to_tensor(image, dtype='float32')

          yield [labels, image] 
    
    return batch_iter(with_charid=with_charid)