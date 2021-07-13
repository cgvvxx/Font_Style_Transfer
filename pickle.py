import glob
import os
import pickle as pickle


def pickle_examples(with_charid=True):
    os.makedirs('./pickledata')
    dir_lst = os.listdir('./train_set')
    c = -1
    with open('./pickledata/train.pickle', 'wb') as ft:
        for font_dir in dir_lst:
            c += 1
            final_dir = './train_set/' + font_dir
            paths = glob.glob(os.path.join(final_dir, "*.png"))
            print('all data num:', len(paths))
            if with_charid:
                print('pickle with charid')
                for p in paths:  # p는 해당 png 파일 명
                    label = c  # 순서대로
                    _idx = os.path.basename(p).rfind('_') + 1
                    charid = int(os.path.basename(p)[_idx:].split(".")[0], 16)  # 0056 -> 86

                    with open(p, 'rb') as f:
                        img_bytes = f.read()
                        example = (label, charid, img_bytes)
                        pickle.dump(example, ft)

    return print("We've just pickled pickle!")
