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
import h5py

from skimage import color, exposure, transform
from scipy import ndimage
from scipy.ndimage.filters import median_filter
from img_util import preprocess_image

import nets


# from 6o6o's fork. https://github.com/6o6o/chainer-fast-neuralstyle/blob/master/generate.py
def original_colors(original, stylized):
    # Histogram normalization in v channel
    ratio=0.9

    hsv = color.rgb2hsv(original/255)
    hsv_s = color.rgb2hsv(stylized/255)

    hsv[:,:,0] = (ratio* hsv_s[:,:,0]) + (1-ratio)*hsv [:,:,0]
    hsv[:,:,2] = (ratio* hsv_s[:,:,2]) + (1-ratio)*hsv [:,:,2]
    img = color.hsv2rgb(hsv)    
    return img


def load_weights(model,file_path):
    f = h5py.File(file_path)

    #print [f[x].attrs for x in f.attrs['layer_names']]
    

    layer_names = [name for name in f.attrs['layer_names']]

    for i, layer in enumerate(model.layers[:31]):
        g = f[layer_names[i]]
        weights = [g[name] for name in g.attrs['weight_names']]
        layer.set_weights(weights)


    # for k in range(f.attrs['nb_layers']):
    #     if k >= len(model.layers) - 1:
    #         # we don't look at the last two layers in the savefile (fully-connected and activation)
    #         break
    #     g = f['layer_{}'.format(k)]
    #     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    #     layer = model.layers[k]

    #     layer.set_weights(weights)

    f.close()
    

    print('Pretrained Model weights loaded.')        

def main(args):
    style= "la_muse" #args.style
    img_width = img_height =  args.image_size
    output_file =args.output
    input_file = "asus-zenbo-christmas.jpg" #args.input

        
    net = nets.image_transform_net(img_width,img_height)
    model = nets.loss_net(net.output,net.input,img_width,img_height,"",0,0)
 
    model.summary()

    model.compile(Adam(),  dummy_loss)  # Dummy loss since we are learning from regularizes

    model.load_weights("pretrained/"+style+'_weights.h5',by_name=False)
    
    #load_weights(model, "pretrained/" + (style+'_weights.h5'))

    x = preprocess_image(input_file,img_width,img_height)

    t1 = time.time()
    y = net.predict(x)[0]
    print("process: %s" % (time.time() -t1))

    imsave('%s_%s.png' % (output_file,"org"), y)
 
    imsave('%s_%s.png' % (output_file,"filter"), median_filter(y,5))


    imsave('%s_%s.png' % (output_file,"color"), original_colors(x[0],median_filter(y,3)))

    

         

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time style transfer')

    parser.add_argument('--style', '-s', type=str, required=True,
                        help='style image file name without extension')

    parser.add_argument('--input', '-i', default=None, required=True,type=str,
                        help='input file name')

    parser.add_argument('--output', '-o', default=None, required=True,type=str,
                        help='output file name without extension')

    parser.add_argument('--image_size', default=256, type=int)

    args = parser.parse_args()
    main(args)
