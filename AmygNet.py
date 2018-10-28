from __future__ import absolute_import, print_function

from niftynet.layer import layer_util
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.crop import CropLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.layer.upsample import UpSampleLayer
from niftynet.network.base_net import BaseNet


class dilated_deepmedic(BaseNet):
    """
    reimplementation of DeepMedic:
        Kamnitsas et al., "Efficient multi-scale 3D CNN with fully connected
        CRF for accurate brain lesion segmentation", MedIA '17
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name="dilated_deepmedic"):

        super(dilated_deepmedic, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.crop_diff = 16
        self.conv_features = [30, 30, 40, 40, 40, 40, 50, 50, 50,50]
        self.fc_features = [150, 150]
        self.conv_classification = [num_classes]

        self.layers_dilation1 = [30, 50]
        self.layers_dilation2 = [30, 40, 40, 50]
        self.layers_dilation4 = [40, 50]
        self.layers_dilation8 = [40]

    def layer_op(self, images, is_training, keep_prob=0.5, layer_id=-1, **unused_kwargs):
        # image_size is defined as the largest context, then:
        #   downsampled path size: image_size / d_factor
        #   downsampled path output: image_size / d_factor - 16
                # to make sure same size of feature maps from both pathways:
        #   normal path size: (image_size / d_factor - 16) * d_factor + 16
        #   normal path output: (image_size / d_factor - 16) * d_factor

        # where 16 is fixed by the receptive field of conv layers
        # TODO: make sure label_size = image_size/d_factor - 16

        # image_size has to be an odd number and divisible by 3 and
        # smaller than the smallest image size of the input volumes

        # label_size should be (image_size/d_factor - 16) * d_factor


        # crop 26x26x26 from 56x56x56
        crop_op = CropLayer(border=self.crop_diff, name='cropping_input')
        normal_path = crop_op(images)
        print(crop_op)

        # dilated pathway

        # dilation rate = 1
        dilated_block = ConvolutionalLayer(
                                        n_output_chns=self.layers_dilation1[0],
                                        kernel_size=3,
                                        padding='VALID',
                                        dilation=1,
                                        w_initializer=self.initializers['w'],
                                        w_regularizer=self.regularizers['w'],
                                        acti_func=self.acti_func,
                                        name='dilated_conv_{}'.format(self.layers_dilation1[0]))

        dilated_path = dilated_block(images, is_training)
        print(dilated_block)

        # dilation rate = 2
        dilated_block = ConvolutionalLayer(
                                n_output_chns=self.layers_dilation2[0],
                                kernel_size=3,
                                padding='VALID',
                                dilation=2,
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='dilated_conv_{}'.format(self.layers_dilation2[0]))

        dilated_path = dilated_block(dilated_path, is_training)
        print(dilated_block)

        #dilation rate = 4
        dilated_block = ConvolutionalLayer(
                                n_output_chns=self.layers_dilation4[0],
                                kernel_size=3,
                                dilation=4,
                                padding='VALID',
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='dilated_conv_{}'.format(self.layers_dilation4[0]))

        dilated_path = dilated_block(dilated_path, is_training)
        print(dilated_block)

        #dilation rate = 2
        dilated_block = ConvolutionalLayer(
                                n_output_chns=self.layers_dilation2[1],
                                kernel_size=3,
                                dilation=2,
                                padding='VALID',
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='dilated_conv_{}'.format(self.layers_dilation2[1]))

        dilated_path = dilated_block(dilated_path, is_training)
        print(dilated_block)

        #dilation rate = 8
        dilated_block = ConvolutionalLayer(
                                n_output_chns=self.layers_dilation8[0],
                                kernel_size=3,
                                dilation=8,
                                padding='VALID',
                                w_initializer=self.initializers['w'],
                                acti_func=self.acti_func,
                                name='dilated_conv_{}'.format(self.layers_dilation8[0]))

        dilated_path= dilated_block(dilated_path, is_training)
        print(dilated_block)

        #dilation rate = 2
        dilated_block = ConvolutionalLayer(
                                n_output_chns=self.layers_dilation2[2],
                                kernel_size=3,
                                dilation=2,
                                padding='VALID',
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='dilated_conv_{}'.format(self.layers_dilation2[2]))

        dilated_path = dilated_block(dilated_path, is_training)
        print(dilated_block)
        #dilation rate = 4
        dilated_block = ConvolutionalLayer(
                                n_output_chns=self.layers_dilation4[1],
                                kernel_size=3,
                                dilation=4,
                                padding='VALID',
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='dilated_conv_{}'.format(self.layers_dilation4[1]))

        dilated_path = dilated_block(dilated_path, is_training)
        print(dilated_block)

        #dilation rate = 2
        dilated_block = ConvolutionalLayer(
                                n_output_chns=self.layers_dilation2[3],
                                kernel_size=3,
                                dilation=2,
                                padding='VALID',
                                w_initializer=self.initializers['w'],
                                w_regularizer=self.regularizers['w'],
                                acti_func=self.acti_func,
                                name='dilated_conv_{}'.format(self.layers_dilation2[3]))

        dilated_path = dilated_block(dilated_path, is_training)
        
        #dilation rate = 1
        dilated_block = ConvolutionalLayer(
                                     n_output_chns=self.layers_dilation1[1],
                                        kernel_size=3,
                                        dilation=1,
                                        padding='VALID',
                                        w_initializer=self.initializers['w'],
                                        w_regularizer=self.regularizers['w'],
                                        acti_func=self.acti_func,
                                        name='dilated_conv_{}'.format(self.layers_dilation1[1]))
        dilated_path = dilated_block(dilated_path, is_training)
        print(dilated_block)

        count_1 = 0

        for n_features in self.conv_features:
            count_1 = count_1 + 1

            # normal pathway convolutions
            conv_path_1 = ConvolutionalLayer(
                n_output_chns=n_features,
                kernel_size=3,
                padding='VALID',
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func,
                name='normal_conv_{}'.format(n_features))

            if count_1 > 4:
               normal_path = conv_path_1(normal_path, is_training, keep_prob=1)
#               print('###########################################keep_prob: ',keep_prob)
            else:
               normal_path = conv_path_1(normal_path, is_training)
            print(conv_path_1)
            


        # upsampling the downsampled pathway
      #  dilated_path = UpSampleLayer('REPLICATE',
       #                               kernel_size=2,
        #                              stride=2)(dilated_path)
       # print(UpSampleLayer)

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
            print('###########################################keep_prob: ',keep_prob)
            print(conv_fc)

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

              
        
