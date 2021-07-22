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
class PickledImageProvider(object):
    def __init__(self, obj_path, verbose):
        self.obj_path = obj_path
        self.verbose = verbose
        self.examples = self.load_pickled_examples()

    def load_pickled_examples(self):
        with open(self.obj_path, "rb") as of:
            examples = list()
            while True:
                try:
                    e = pickle.load(of)
                    examples.append(e)
                except EOFError:
                    break
                except Exception:
                    pass
            if self.verbose:
                print("unpickled total %d examples" % len(examples))
            return examples


class TrainDataProvider(object):
    def __init__(self, data_dir, train_name, filter_by_font=None, filter_by_charid=None, verbose=True):
        self.data_dir = data_dir
        self.train_name = train_name
        self.filter_by_font = filter_by_font
        self.filter_by_charid = filter_by_charid
        self.train_path = os.path.join(self.data_dir, self.train_name)
        self.train = PickledImageProvider(self.train_path, verbose)

    def get_train_iter(self, batch_size, shuffle=True, with_charid=False):
        training_examples = self.train.examples[:]
        if shuffle:
            np.random.shuffle(training_examples)

        if with_charid:
            return get_batch_iter(training_examples, batch_size, augment=True, with_charid=True)
        else:
            return get_batch_iter(training_examples, batch_size, augment=True)

    def compute_total_batch_num(self, batch_size):
        return int(np.ceil(len(self.train.examples) / float(batch_size)))


def init_embedding(font_total_num, embedding_dim, stddev=0.01):
    embedding = tf.random.normal([font_total_num, embedding_dim], mean=0, stddev=stddev)
    embedding = tf.reshape(embedding, shape=[font_total_num, 1, 1, embedding_dim])
    return embedding


def embedding_lookup(embeddings, embedding_ids):
    batch_size = len(embedding_ids)
    embedding_dim = embeddings.shape[3]
    local_embeddings = []
    for id_ in embedding_ids:
        local_embeddings.append(embeddings[id_].numpy())

    local_embeddings = tf.convert_to_tensor(np.array(local_embeddings))
    local_embeddings = tf.reshape(local_embeddings, [batch_size, 1, 1, embedding_dim])
    return local_embeddings