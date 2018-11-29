# By Yilin Liu, 2018
from __future__ import absolute_import, print_function

from niftynet.layer import layer_util
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.crop import CropLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet


class AmygNet(BaseNet):
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
                 name="AmygNet"):

        super(AmygNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.crop_diff = 16
        self.dilation_rates = [1,2,4,2,8,2,4,2,1]
        self.conv1_features = [30, 30, 40, 40, 40, 40, 50, 50, 50, 50]
        self.conv2_features = [30, 30, 40, 40, 40, 40, 50, 50, 50]
        self.fc_features = [150, 150]
        self.conv_classification = [num_classes]



    def layer_op(self, images, is_training, keep_prob=0.5, layer_id=-1, **unused_kwargs):

        # crop 27x27x27 from 59x59x59
        crop_op = CropLayer(border=self.crop_diff, name='cropping_input')
        normal_path = crop_op(images)
        dilated_path = images
        print(crop_op)

        # dilated pathway

        # dilation rate: 1,2,4,2,8,2,4,2,1
        for n_features, rate in zip(self.conv2_features,self.dilation_rates):
            dilated_block = ConvolutionalLayer(
                                        n_output_chns=n_features,
                                        kernel_size=3,
                                        padding='VALID',
                                        dilation=rate,
                                        w_initializer=self.initializers['w'],
                                        w_regularizer=self.regularizers['w'],
                                        acti_func=self.acti_func,
                                        name='dilated_conv_{}'.format(n_features))

            dilated_path = dilated_block(dilated_path, is_training)
            print(dilated_block)
   
        # normal pathway

        for n_features in self.conv1_features:

            # normal pathway convolutions
            conv_path_1 = ConvolutionalLayer(
                n_output_chns=n_features,
                kernel_size=3,
                padding='VALID',
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func,
                name='normal_conv_{}'.format(n_features))

            normal_path = conv_path_1(normal_path, is_training)
            print(conv_path_1)


        # concatenate both pathways
        output_tensor = ElementwiseLayer('CONCAT')(normal_path, dilated_path)

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

