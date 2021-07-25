from model import Encoder, Decoder, Discriminator, Generator
from load_data import TrainDataProvider
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse
import datetime
import glob
import os
import time


class Trainer:

    def __init__(self, opt):

        self.data_dir = opt.data_dir
        self.batch_size = opt.batch_size
        self.img_size = opt.img_size
        self.train_name = opt.train_name
        self.embedding_name = opt.embedding_name

        self.embeddings = pd.read_pickle(os.path.join(self.data_dir, self.embedding_name))
        self.embedding_num = self.embeddings.shape[0]
        self.embedding_dim = self.embeddings.shape[3]

        self.fonts_num = len(self.embeddings)

        self.data_provider = TrainDataProvider(self.data_dir, self.train_name)
        self.total_batches = self.data_provider.compute_total_batch_num(self.batch_size)
        print("total batches:", self.total_batches)

    def gd_loss(self, real_source, real_target, fake_target, encoded_source, embedding_ids,
                En, Dis, L1_penalty, Lconst_penalty, Lcat_penalty):

        l1_criterion = tf.keras.losses.MeanAbsoluteError()
        bce_criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        mse_criterion = tf.keras.losses.MeanSquaredError()

        mae = tf.keras.losses.MeanAbsoluteError()

        real_TS = tf.concat([real_source, real_target], axis=3)
        fake_TS = tf.concat([real_source, fake_target], axis=3)

        real_score, real_score_logit, real_cat_logit = Dis(real_TS)
        fake_score, fake_score_logit, fake_cat_logit = Dis(fake_TS)

        # G_1. constant loss
        encoded_fake = En(fake_target)[0]
        const_loss = Lconst_penalty * mse_criterion(encoded_source, encoded_fake)

        # G_2, D_1. category loss
        real_category = tf.convert_to_tensor(np.eye(self.fonts_num)[embedding_ids])
        real_category_loss = Lcat_penalty * bce_criterion(real_category, real_cat_logit)
        fake_category_loss = Lcat_penalty * bce_criterion(real_category, fake_cat_logit)
        category_loss = 0.5 * (real_category_loss + fake_category_loss)

        # D_2. binary loss - T/F
        one_labels = tf.ones([self.batch_size, 1])
        zero_labels = tf.zeros([self.batch_size, 1])

        real_binary_loss = bce_criterion(real_score, one_labels)
        fake_binary_loss = bce_criterion(fake_score, zero_labels)
        binary_loss = 10 * (real_binary_loss + fake_binary_loss)

        # G_3. L1 loss between real and fake images
        l1_loss = L1_penalty * l1_criterion(real_target, fake_target)

        # G_4. cheat loss for generator to fool discriminator
        cheat_loss = 10 * bce_criterion(fake_score, one_labels)

        # g_loss, d_loss
        g_loss = const_loss + fake_category_loss + l1_loss + cheat_loss
        d_loss = category_loss + binary_loss
        return g_loss, d_loss, [cheat_loss, l1_loss, fake_category_loss, const_loss, binary_loss, category_loss]

    def generate_images(self, real_src, real_tar, En, De, embeddings, embedding_ids):

        fake_tar, encoded_source, encode_layers = Generator(real_src, En, De, embeddings, embedding_ids)

        plt.figure(figsize=(15, 15))
        display_list = [real_src[0], real_tar[0], fake_tar[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(title[i])
            img = tf.concat([display_list[i]] * 3, axis=2)
            plt.imshow(img * 0.5 + 0.5)
            plt.axis('off')
        plt.show()

    def train_step(self, source, target, epoch, embeddings, embedding_ids, L1_penalty, Lconst_penalty, Lcat_penalty,
                   En, De, Dis, g_optimizer, d_optimizer, summary_writer):

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_target, encoded_source, encode_layers = Generator(source, En, De, embeddings, embedding_ids)
            gen_loss, disc_loss, loss_list = self.gd_loss(source, target, fake_target, encoded_source, embedding_ids,
                                                          En, Dis, L1_penalty, Lconst_penalty, Lcat_penalty)

        ge_trainable_variables = En.trainable_variables + De.trainable_variables

        generator_gradients = gen_tape.gradient(gen_loss,
                                                ge_trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     Dis.trainable_variables)

        g_optimizer.apply_gradients(zip(generator_gradients,
                                        ge_trainable_variables))
        d_optimizer.apply_gradients(zip(discriminator_gradients,
                                        Dis.trainable_variables))

        with summary_writer.as_default():
            tf.summary.scalar('G_gen_loss', gen_loss, step=epoch)
            tf.summary.scalar('D_disc_loss', disc_loss, step=epoch)
            tf.summary.scalar('G1_const_loss', loss_list[3], step=epoch)
            tf.summary.scalar('G2_fake_category_loss', loss_list[2], step=epoch)
            tf.summary.scalar('G3_l1_loss', loss_list[1], step=epoch)
            tf.summary.scalar('G4_cheat_loss', loss_list[0], step=epoch)
            tf.summary.scalar('D1_category_loss', loss_list[5], step=epoch)
            tf.summary.scalar('D2_binary_loss', loss_list[4], step=epoch)

    def train(self, max_epoch, schedule, lr=0.001, fine_tune=False, flip_labels=False,
              restore=None, ckpt_dir=None, with_charid=True, model_save_step=1):
        max_epoch = opt.max_epoch
        schedule = opt.schedule
        lr = opt.learning_rate

        if not fine_tune:
            L1_penalty, Lconst_penalty, Lcat_penalty = 100, 50, 100
        else:
            L1_penalty, Lconst_penalty, Lcat_penalty = 500, 1000, 500

        En = Encoder()
        De = Decoder()
        Dis = Discriminator(cat_num=self.fonts_num)

        g_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)
        d_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

        log_dir = "./logs/"
        summary_writer = tf.summary.create_file_writer(log_dir + datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))

        checkpoint_dir = './ckpt'
        checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer, discriminator_optimizer=d_optimizer,
                                         encoder=En, decoder=De, discriminator=Dis)

        if restore:
            files = glob.glob(os.path.join(ckpt_dir, '*'))
            ckpt_list = [file[:file.rfind('.')] for file in files if file.endswith(".index")]

            prev_epoch = int(ckpt_list[-1].split('-')[0][-5:-2])

            g_optimizer = tf.keras.optimizers.Adam()
            d_optimizer = tf.keras.optimizers.Adam()

            En = Encoder()
            De = Decoder()
            Dis = Discriminator(cat_num=self.fonts_num)

            checkpoint = tf.train.Checkpoint(generator_optimizer=g_optimizer, discriminator_optimizer=d_optimizer,
                                       encoder=En, decoder=De, discriminator=Dis)

            checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))
            print("%d epoch trained model has restored" % prev_epoch)
        else:
            prev_epoch = 0
            print("New model training start!!")

        start = time.time()
        for epoch in range(max_epoch - prev_epoch):

            epoch += prev_epoch
            train_batch_iter = self.data_provider.get_train_iter(self.batch_size, with_charid=with_charid)

            if (epoch + 1) % schedule == 0:
                updated_lr = max(lr / 2, 0.0001)
                g_optimizer = tf.keras.optimizers.Adam(updated_lr, beta_1=0.5)
                d_optimizer = tf.keras.optimizers.Adam(updated_lr, beta_1=0.5)
                if lr != updated_lr:
                    print("decay learning rate from %.5f to %.5f" % (lr, updated_lr))
                lr = updated_lr

                updated_Lcat = min(10000, 2 * Lcat_penalty)
                if Lcat_penalty != updated_Lcat:
                    print("increase category penalty from %.5f to %.5f" % (Lcat_penalty, updated_Lcat))
                Lcat_penalty = updated_Lcat

            perc = 0
            for i, batch in enumerate(train_batch_iter):

                if i == 0:
                    val_embedding_ids = batch[0]
                    val_source = tf.reshape(batch[2][:, :, :, 0], [self.batch_size, self.img_size, self.img_size, 1])
                    val_target = tf.reshape(batch[2][:, :, :, 1], [self.batch_size, self.img_size, self.img_size, 1])

                    print('\n %d EPOCHS :' % (epoch + 1))
                    self.generate_images(val_source, val_target, En, De, self.embeddings, val_embedding_ids)

                    print("Learning Rate : {0}, L1_penalty : {1}, Lconst_penalty : {2}, Lcat_penalty : {3} \n".format(
                        lr, L1_penalty, Lconst_penalty, Lcat_penalty))

                perc_check = perc
                perc = (i / self.total_batches) * 100
                if (perc // 3) - (perc_check // 3) != 0:
                    print('....', end='')

                if with_charid:
                    font_ids, char_ids, batch_images = batch
                else:
                    font_ids, batch_images = batch

                embedding_ids = font_ids

                if flip_labels:
                    np.random.shuffle(embedding_ids)

                real_target = tf.reshape(batch_images[:, :, :, 1], [self.batch_size, self.img_size, self.img_size, 1])
                real_source = tf.reshape(batch_images[:, :, :, 0], [self.batch_size, self.img_size, self.img_size, 1])

                self.train_step(real_source, real_target, epoch, self.embeddings, embedding_ids,
                                L1_penalty, Lconst_penalty, Lcat_penalty, En, De, Dis,
                                g_optimizer, d_optimizer, summary_writer)

            if (epoch + 1) % model_save_step == 0:
                checkpoint_prefix = os.path.join(checkpoint_dir, f"ckpt_{epoch + 1}EP")
                checkpoint.save(file_prefix=checkpoint_prefix)

            if epoch == 0:
                print("\n It takes about %.0fs for 1 EPOCH" % float(time.time() - start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help='the path Where pickledata exist')
    parser.add_argument('--embedding_name', required=True, help='the name of embedding file')
    parser.add_argument('--train_name', required=True, help='the name of training file')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--img_size', default=128, type=int, help='image size')
    parser.add_argument('--max_epoch', default=1000, type=int, help='maximum of training epoch')
    parser.add_argument('--schedule', default=50, type=int, help='the term of weight updating')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--fine_tune', action='store_true', help='Whether fine tuning')

    opt = parser.parse_args()

    trainer = Trainer(opt)
    trainer.train(opt)


