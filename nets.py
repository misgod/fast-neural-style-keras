from keras.layers import Input, merge
from keras.models import Model,Sequential
from layers import VGGNormalize,ReflectionPadding2D,Denormalize,conv_bn_relu,res_conv,dconv_bn_nolinear
from loss import StyleReconstructionRegularizer,FeatureReconstructionRegularizer,TVRegularizer
from keras import backend as K
from VGG16 import vgg16
import img_util



style_weight=5.
content_weight=1.
tv_weight=1e-4
style_image_path = "images/style/starry_night.jpg"
img_width=256
img_height=256

def image_transform_net():
    x = Input(shape=(256,256,3))
    a = ReflectionPadding2D(padding=(40,40),input_shape=(256,256,3))(x)
    a = conv_bn_relu(32, 9, 9, stride=(1,1))(a)
    a = conv_bn_relu(64, 9, 9, stride=(2,2))(a)
    a = conv_bn_relu(128, 3, 3, stride=(2,2))(a)
    for i in range(5):
        a = res_conv(128,3,3)(a)
    a = dconv_bn_nolinear(64,3,3,output_shape=(1,128,128,64))(a)
    a = dconv_bn_nolinear(32,3,3,output_shape=(1,256,256,32))(a)
    a = dconv_bn_nolinear(3,9,9,stride=(1,1),activation="tanh",output_shape=(1,256,256,3))(a)
    # Scale output to range [0, 255] via custom Denormalize layer
    y = Denormalize(name='transform_output')(a)
    
    model = Model(input=x, output=y)
    add_total_variation_loss(model.layers[-1])
    return model 




def loss_net(x_in, trux_x_in):
    # Append the initial input to the FastNet input to the VGG inputs
    x = merge([x_in, trux_x_in], mode='concat', concat_axis=0)

    # Normalize the inputs via custom VGG Normalization layer
    x = VGGNormalize(name="vgg_normalize")(x)

    vgg = vgg16(include_top=False,input_tensor=x)

    vgg_output_dict = dict([(layer.name, layer.output) for layer in vgg.layers[-18:]])
    vgg_layers = dict([(layer.name, layer) for layer in vgg.layers[-18:]])
    
    #add_style_loss(vgg,style_image_path , vgg_layers, vgg_output_dict)    
    add_content_loss(vgg_layers,vgg_output_dict)    
    
    # Freeze all VGG layers
    for layer in vgg.layers[-19:]:
        layer.trainable = False

    return vgg
    


def add_style_loss(vgg,style_image_path,vgg_layers,vgg_output_dict): 
    style_img = img_util.preprocess_image(style_image_path, img_width, img_height)
    print('Getting style features from VGG network.')

    style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3']

    style_layer_outputs = []

    for layer in style_layers:
        style_layer_outputs.append(vgg_output_dict[layer])

    vgg_style_func = K.function([vgg.layers[-19].input], style_layer_outputs)    
        
    style_features = vgg_style_func([style_img])

    # Style Reconstruction Loss
    for i, layer_name in enumerate(style_layers):
        layer = vgg_layers[layer_name]
        
        feature_var = K.variable(value=style_features[i][0])
        style_loss = StyleReconstructionRegularizer(
                            style_feature_target=feature_var,
                            weight=style_weight)(layer)

        layer.add_loss(style_loss)

def add_content_loss(vgg_layers,vgg_output_dict):    
    # Feature Reconstruction Loss
    content_layer = 'block3_conv3'
    content_layer_output = vgg_output_dict[content_layer]

    layer = vgg_layers[content_layer]
    content_regularizer = FeatureReconstructionRegularizer(
                                       weight=content_weight)(layer)
    layer.add_loss(content_regularizer)    


def add_total_variation_loss(transform_output_layer):
    # Total Variation Regularization
    layer = transform_output_layer  # Output layer
    tv_regularizer = TVRegularizer(weight=tv_weight)(layer)
    layer.add_loss(tv_regularizer)    

