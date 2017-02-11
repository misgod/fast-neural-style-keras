from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import merge
from keras.engine import InputSpec
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Deconvolution2D, Convolution2D,UpSampling2D,Cropping2D
from VGG16 import vgg16
from keras.applications.vgg16 import preprocess_input
import numpy as np
import tensorflow as tf




def conv_bn_relu(nb_filter, nb_row, nb_col,stride):   
    def conv_func(x):
        x = Convolution2D(nb_filter, nb_row, nb_col, subsample=stride,border_mode='same')(x)
        x = BatchNormalization(mode=1)(x)
        x = Activation('relu')(x)
        return x
    return conv_func    



#https://keunwoochoi.wordpress.com/2016/03/09/residual-networks-implementation-on-keras/
def res_conv(nb_filter, nb_row, nb_col,stride=(1,1)):   
    def _res_func(x):
        identity = Cropping2D(cropping=((2,2),(2,2)))(x)  
        
        a = Convolution2D(nb_filter, nb_row, nb_col, subsample=stride, border_mode='valid')(x)
        a = BatchNormalization(mode=1)(a)
        a = Activation('relu')(a)
        a = Convolution2D(nb_filter, nb_row, nb_col, subsample=stride, border_mode='valid')(a)
        y = BatchNormalization(mode=1)(a)

        return  merge([identity, y], mode='sum')
    
    return _res_func    

    
def dconv_bn_relu(nb_filter, nb_row, nb_col,stride=(2,2)):   
    def dconv_bn_relu(x):
        #TODO: Deconvolution2D
        x = UpSampling2D(size=stride)(x)
        x = Convolution2D(nb_filter,nb_row, nb_col, border_mode='same')(x)
        x = Activation('relu')(x)
        return x
    return dconv_bn_relu        


class Denormalize(Layer):
    '''
    Custom layer to denormalize the final Convolution layer activations (tanh)
    Since tanh scales the output to the range (-1, 1), we add 1 to bring it to the
    range (0, 2). We then multiply it by 127.5 to scale the values to the range (0, 255)
    '''

    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Scales the tanh output activations from previous layer (-1, 1) to the
        range (0, 255)
        '''

        return (x + 1) * 127.5

    def get_output_shape_for(self, input_shape):
        return input_shape


class VGGNormalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG network.
    '''

    def __init__(self, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        # No exact substitute for set_subtensor in tensorflow
        # So we subtract an approximate value
        # x = preprocess_input(x)
        x -= 120
       
        return x
   

    def get_output_shape_for(self, input_shape):
        return input_shape




class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), dim_ordering='default', **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()

        self.padding = padding
        if isinstance(padding, dict):
            if set(padding.keys()) <= {'top_pad', 'bottom_pad', 'left_pad', 'right_pad'}:
                self.top_pad = padding.get('top_pad', 0)
                self.bottom_pad = padding.get('bottom_pad', 0)
                self.left_pad = padding.get('left_pad', 0)
                self.right_pad = padding.get('right_pad', 0)
            else:
                raise ValueError('Unexpected key found in `padding` dictionary. '
                                 'Keys have to be in {"top_pad", "bottom_pad", '
                                 '"left_pad", "right_pad"}.'
                                 'Found: ' + str(padding.keys()))
        else:
            padding = tuple(padding)
            if len(padding) == 2:
                self.top_pad = padding[0]
                self.bottom_pad = padding[0]
                self.left_pad = padding[1]
                self.right_pad = padding[1]
            elif len(padding) == 4:
                self.top_pad = padding[0]
                self.bottom_pad = padding[1]
                self.left_pad = padding[2]
                self.right_pad = padding[3]
            else:
                raise TypeError('`padding` should be tuple of int '
                                'of length 2 or 4, or dict. '
                                'Found: ' + str(padding))

        if dim_ordering not in {'tf'}:
            raise ValueError('dim_ordering must be in {tf}.')
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=4)] 


    def call(self, x, mask=None):
        top_pad=self.top_pad
        bottom_pad=self.bottom_pad
        left_pad=self.left_pad
        right_pad=self.right_pad        
        
        paddings = [[0,0],[left_pad,right_pad],[top_pad,bottom_pad],[0,0]]

        
        return tf.pad(x,paddings, mode='REFLECT', name=None)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'tf':
            rows = input_shape[1] + self.top_pad + self.bottom_pad if input_shape[1] is not None else None
            cols = input_shape[2] + self.left_pad + self.right_pad if input_shape[2] is not None else None

            return (input_shape[0],
                    rows,
                    cols,
                    input_shape[3])
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
            
    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))           
        