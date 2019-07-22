# By Yilin Liu, 2018
from __future__ import absolute_import, print_function

from niftynet.layer import layer_util
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.crop import CropLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet


class Multi_AmygNet(BaseNet):
    """
    implementation
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name="Multi_AmygNet"):

        super(AmygNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.crop_diff = 16
        #self.dilation_rates = [1,2,4,2,8,2,4,2,1]
        self.T1_features = [30, 30, 40, 40, 40, 40, 50, 50, 50, 50]
        #self.T2_features = [30, 30, 40, 40, 40, 40, 50, 50, 50, 50]
        #self.fc_features = [150]
        self.conv_classification = [num_classes]
        
        #self.c-SE = ChannelSELayer()
        #self.s-SE = SpatialSELayer()
        #self.cs-SE = ChannelSpatialSELayer()


    def layer_op(self, images, is_training, keep_prob=0.5, layer_id=-1, **unused_kwargs):

        # crop 27x27x27 from 59x59x59
        #crop_op = CropLayer(border=self.crop_diff, name='cropping_input')
        T1_path = images
       # T2_path = crop_op(images)
        print(crop_op)

        # T1 pathway
        for n_features in self.T1_features:

            # T1 pathway convolutions
            T1_path_1 = ConvolutionalLayer(
                n_output_chns=n_features,
                kernel_size=3,
                padding='VALID',
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func,
                name='normal_conv_{}'.format(n_features))
            
            c-SE = ChannelSELayer()

            T1_path = c-SE(T1_path_1(T1_path, is_training))
            print(T1_path_1)
            print(c-SE)
   '''
        # T2 pathway
        for n_features in self.conv2_features:

            # T2 pathway convolutions
            T2_path_1 = ConvolutionalLayer(
                n_output_chns=n_features,
                kernel_size=3,
                padding='VALID',
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func,
                name='normal_conv_{}'.format(n_features))

            T2_path = T2_path_1(T2_path, is_training)
            print(T2_path_1)
    '''

        # concatenate both pathways
        #output_tensor = ElementwiseLayer('CONCAT')(T1_path, T2_path)
        # weighted concatenation - with attention mechanisms
        #output_tensor = self.c-SE(output_tensor)

        # 1x1x1 convolution layer
        for n_features in self.fc_features:
            conv_fc = ConvolutionalLayer(
                n_output_chns=n_features,
                kernel_size=1,
                acti_func=self.acti_func,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='conv_1x1x1_{}'.format(n_features))

            output_tensor = conv_fc(output_tensor, is_training, keep_prob=keep_prob)
            print('#----------------------------------- keep_prob: ', keep_prob)
            print(conv_fc)
        
        # classification layer
        for n_features in self.conv_classification:
            conv_classification = ConvolutionalLayer(
                n_output_chns=n_features,
                kernel_size=1,
                acti_func=None,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='conv_1x1x1_{}'.format(n_features))
            output_tensor = conv_classification(output_tensor, is_training)
            print(conv_classification)

        return output_tensor
