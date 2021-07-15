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



def tight_crop_image(img, verbose=False, resize_fix=False):
    img_size = img.shape[0]
    full_white = img_size
    col_sum = np.where(full_white - np.sum(img, axis=0) > 1)
    row_sum = np.where(full_white - np.sum(img, axis=1) > 1)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    cropped_image = img[y1:y2, x1:x2]
    print('cropped_image shape :', cropped_image.shape)
    cropped_image_size = cropped_image.shape
    
    if verbose:
        print('(left x1, top y1):', (x1, y1))
        print('(right x2, bottom y2):', (x2, y2))
        print('cropped_image size:', cropped_image_size)
        
    if type(resize_fix) == int:
        origin_h, origin_w = cropped_image.shape
        if origin_h > origin_w:
            resize_w = int(origin_w * (resize_fix / origin_h))
            resize_h = resize_fix
        else:
            resize_h = int(origin_h * (resize_fix / origin_w))
            resize_w = resize_fix
        if verbose:
            print('resize_h:', resize_h)
            print('resize_w:', resize_w, \
                  '[origin_w %d / origin_h %d * target_h %d]' % (origin_w, origin_h, target_h))
        
        # resize
    
        cropped_image = np.array(Image.fromarray(cropped_image).resize([resize_h, resize_w])) # 수정
        cropped_image = normalize_image(cropped_image)
        
        cropped_image_size = cropped_image.shape
        if verbose:
            print('resized_image size:', cropped_image_size)
        
    elif type(resize_fix) == float:
        origin_h, origin_w = cropped_image.shape
        resize_h, resize_w = int(origin_h * resize_fix), int(origin_w * resize_fix)
        if resize_h > 120:
            resize_h = 120
            resize_w = int(resize_w * 120 / resize_h)
        if resize_w > 120:
            resize_w = 120
            resize_h = int(resize_h * 120 / resize_w)
        if verbose:
            print('resize_h:', resize_h)
            print('resize_w:', resize_w)
        
        # resize
        cropped_image = np.array(Image.fromarray(cropped_image).resize([resize_h, resize_w]))
        

        cropped_image = normalize_image(cropped_image)
        cropped_image_size = cropped_image.shape
        if verbose:
            print('resized_image size:', cropped_image_size)
    
    return cropped_image


def add_padding(img, image_size=128, verbose=False, pad_value=None):
    height, width = img.shape
    if not pad_value:
        pad_value = img[0][0]
    if verbose:
        print('original cropped image size:', img.shape)
    
    # Adding padding of x axis - left, right
    pad_x_width = (image_size - width) // 2
    pad_x = np.full((height, pad_x_width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_x, img), axis=1)
    img = np.concatenate((img, pad_x), axis=1)
    
    width = img.shape[1]

    # Adding padding of y axis - top, bottom
    pad_y_height = (image_size - height) // 2
    pad_y = np.full((pad_y_height, width), pad_value, dtype=np.float32)
    img = np.concatenate((pad_y, img), axis=0)
    img = np.concatenate((img, pad_y), axis=0)
    
    # Match to original image size
    width = img.shape[1]
    if img.shape[0] % 2:
        pad = np.full((1, width), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=0)
    height = img.shape[0]
    if img.shape[1] % 2:
        pad = np.full((height, 1), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=1)

    if verbose:
        print('final image size:', img.shape)
    
    return img


def centering_image(img, image_size=128, verbose=False, resize_fix=False, pad_value=None):
    if not pad_value:
        pad_value = img[0][0]
    cropped_image = tight_crop_image(img, verbose=verbose, resize_fix=resize_fix)
    centered_image = add_padding(cropped_image, image_size=image_size, verbose=verbose, pad_value=pad_value)
    
    return centered_image

def round_function(i):
    if i < -0.95:
        return -1
    elif i > 0.95:
        return 1
    else:
        return i