import cv2
from tqdm import tqdm
import os
import glob
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def resize():
    font_list = os.listdir("./ttf2png")

    for font in font_list:

        new_path = "./resized_img/" + font[:-4]

        os.makedirs(new_path, exist_ok=True)

        Characters = glob.glob("./ttf2png/{}/*".format(font))

        print(font_list.index(font))

        for image in tqdm(Characters):

            img_path = image

            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

            length_origin = img.shape[0]
            width_origin = img.shape[1]

            portion = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            portion.sort(reverse=True)

            new_ratio = 1

            if max(length_origin, width_origin) > 128:
                for ratio in portion:
                    revised_size = max(length_origin, width_origin) * ratio
                    if revised_size < 128:
                        new_ratio = ratio
                        break

            img_reduced = cv2.resize(img, dsize=(0, 0), fx=new_ratio, fy=new_ratio)

            length = img_reduced.shape[0]
            width = img_reduced.shape[1]

            max_size = 128  # resize 128*128

            if (max_size - length) % 2 == 1:
                top, bottom = int((max_size - length) / 2), int((max_size - length) / 2) + 1
            else:
                top, bottom = int((max_size - length) / 2), int((max_size - length) / 2)

            if (max_size - width) % 2 == 1:
                left, right = int((max_size - width) / 2), int((max_size - width) / 2) + 1
            else:
                left, right = int((max_size - width) / 2), int((max_size - width) / 2)

            img_resized = cv2.copyMakeBorder(img_reduced, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                             value=[255, 255, 255])

            # centerize characters

            dx, dy = 0, -30

            mtrx = np.float32([[1, 0, dx], [0, 1, dy]])

            img_center = cv2.warpAffine(img_resized, mtrx, (128 + dx, 128 + dy), None, cv2.INTER_LINEAR,
                                        cv2.BORDER_CONSTANT, (255, 255, 255))

            img_final = cv2.copyMakeBorder(img_center, 15, 15, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            new_img_path = new_path + "/" + image.split("\\")[-1]

            cv2.imwrite(new_img_path, img_final)


def concat_img():
    base = os.listdir("./resized_img/NotoSans-Regular/")

    train_list = os.listdir("./resized_img/")
    train_list.remove('NotoSans-Regular')

    for font in train_list:

        Characters = os.listdir("./resized_img/" + font)

        os.makedirs("./train_set/" + font, exist_ok=True)

        print(font)

        for Char in tqdm(Characters):
            img_base = cv2.imread("./resized_img/NotoSans-Regular/" + base[Characters.index(Char)], 0)
            img_train = cv2.imread("./resized_img/" + font + "/" + Char, 0)

            img_base = cv2.resize(img_base, (128, 128))
            img_train = cv2.resize(img_train, (128, 128))

            combined_img = cv2.hconcat([img_base, img_train])

            new_img_path = "./train_set/" + font + "/" + Char

            cv2.imwrite(new_img_path, combined_img)
