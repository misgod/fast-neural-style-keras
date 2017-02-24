from keras.layers import Input, merge
from keras.models import Model,Sequential
from layers import VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from loss import dummy_loss,StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
from keras.optimizers import Adam, SGD,Nadam,Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from scipy.misc import imsave
import time
import numpy as np
import argparse

from keras.callbacks import TensorBoard
from scipy import ndimage

import nets



def display_img(i,x,style,is_val=False):
    # save current generated image
    img = x #deprocess_image(x)
    if is_val:
        #img = ndimage.median_filter(img, 3)

        fname = 'images/output/%s_%d_val.png' % (style,i)
    else:
        fname = 'images/output/%s_%d.png' % (style,i)
    imsave(fname, img)
    print('Image saved as', fname)

def get_style_img_path(style):
    return "images/style/"+style+".jpg"


def main(args):
    style_weight= args.style_weight
    content_weight= args.content_weight
    tv_weight= args.tv_weight
    style= args.style
    img_width = img_height =  args.image_size
         
    style_image_path = get_style_img_path(style)

    net = nets.image_transform_net(img_width,img_height,tv_weight)
    model = nets.loss_net(net.output,net.input,img_width,img_height,style_image_path,content_weight,style_weight)
    model.summary()


    nb_epoch = 82785 *2
    train_batchsize =  1
    train_image_path = "images/train/"

    learning_rate = 1e-3 #1e-3
    optimizer = Adam() # Adam(lr=learning_rate,beta_1=0.99)

    model.compile(optimizer,  dummy_loss)  # Dummy loss since we are learning from regularizes

    datagen = ImageDataGenerator()

    dummy_y = np.zeros((train_batchsize,img_width,img_height,3)) # Dummy output, not used since we use regularizers to train

 

    # model.load_weights(style+'_weights.h5',by_name=False)


    i=0
    t1 = time.time()
    for x in datagen.flow_from_directory(train_image_path, class_mode=None, batch_size=train_batchsize,
        target_size=(img_width, img_height), shuffle=False):    
        if i > nb_epoch:
            break

        hist = model.train_on_batch(x, dummy_y)

        if i % 50 == 0:
            print  hist,(time.time() -t1)
            t1 = time.time()

        if i % 200 == 0:
            print("epoc: ", i)
            val_x = net.predict(x)

            display_img(i, x[0], style)
            display_img(i, val_x[0],style, True)
            model.save_weights(style+'_weights.h5')

        i+=1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time style transfer')
        
    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name without extension')
          
    parser.add_argument('--output', '-o', default=None, type=str,
                        help='output model file path without extension')
    parser.add_argument('--tv_weight', default=1e-6, type=float,
                        help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
    parser.add_argument('--content_weight', default=1.0, type=float)
    parser.add_argument('--style_weight', default=8.0, type=float)
    parser.add_argument('--image_size', default=256, type=int)

    args = parser.parse_args()
    main(args)
