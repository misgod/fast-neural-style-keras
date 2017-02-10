from scipy.misc import imread, imresize, imsave, fromimage, toimage
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from PIL import Image
import numpy as np
import os

from keras import backend as K
from keras.preprocessing import image
import numpy as np

def preprocess_image1(image_path,img_nrows,img_ncols):
    img = image.load_img(image_path, target_size=(img_nrows, img_ncols))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
   # img = vgg16.preprocess_input(img)
    return img

# util function to convert a tensor into a valid image


# Util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, img_width=256, img_height=256, load_dims=False, resize=True, size_multiple=4):
    '''
    Preprocess the image so that it can be used by Keras.
    Args:
        image_path: path to the image
        img_width: image width after resizing. Optional: defaults to 256
        img_height: image height after resizing. Optional: defaults to 256
        load_dims: decides if original dimensions of image should be saved,
                   Optional: defaults to False
        vgg_normalize: decides if vgg normalization should be applied to image.
                       Optional: defaults to False
        resize: whether the image should be resided to new size. Optional: defaults to True
        size_multiple: Deconvolution network needs precise input size so as to
                       divide by 4 ("shallow" model) or 8 ("deep" model).
    Returns: an image of shape (3, img_width, img_height) for dim_ordering = "th",
             else an image of shape (img_width, img_height, 3) for dim ordering = "tf"
    '''
    img = imread(image_path, mode="RGB")  # Prevents crashes due to PNG images (ARGB)
    if load_dims:
        global img_WIDTH, img_HEIGHT, aspect_ratio
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = img_HEIGHT / img_WIDTH

    if resize:
        if img_width < 0 or img_height < 0: # We have already loaded image dims
            img_width = (img_WIDTH // size_multiple) * size_multiple # Make sure width is a multiple of 4
            img_height = (img_HEIGHT // size_multiple) * size_multiple # Make sure width is a multiple of 4
        img = imresize(img, (img_width, img_height))

    if K.image_dim_ordering() == "th":
        img = img.transpose((2, 0, 1)).astype(np.float32)
    else:
        img = img.astype(np.float32)

    img = np.expand_dims(img, axis=0)
    return img

def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x
