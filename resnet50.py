<<<<<<< HEAD
=======
"""
A Trainable ResNet-50 Class is defined in this file
Author: Kaihua Tang
"""
import math
import numpy as np
import tensorflow as tf
from functools import reduce


class ResNet:
    def __init__(self, ResNet_npy_path=None, trainable=True, open_tensorboard=False):
        """
        Initialize function
        ResNet_npy_path: If path is not none, loading the model. Otherwise, initialize all parameters at random.
        open_tensorboard: Is Open Tensorboard or not.
        """
        if ResNet_npy_path is not None:
            self.data_dict = np.load(ResNet_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.open_tensorboard = open_tensorboard
        self.is_training = True

    def set_is_training(self, isTrain):
        """
        Set is training bool.
        """
        self.is_training = isTrain

    def build(self, rgb, label_num, last_layer_type="softmax"):
        """
        load variable from npy to build the Resnet or Generate a new one
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """
        # Preprocessing: Turning RGB to BGR - Mean.
        BGR_MEAN = [104.7546, 124.328, 167.1754]
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - BGR_MEAN[0],
            green - BGR_MEAN[1],
            red - BGR_MEAN[2],
        ])
        print(bgr.get_shape().as_list())
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.conv1 = self.conv_layer(bgr, 7, 3, 64, 2, "conv1")
        self.conv_norm_1 = self.batch_norm(self.conv1)
        self.conv1_relu = tf.nn.relu(self.conv_norm_1)

        self.pool1 = self.max_pool(self.conv1_relu, 3, 2, "pool1")
        self.block1_1 = self.res_block_3_layers(self.pool1, [64, 64, 256], "block1_1", True)
        self.block1_2 = self.res_block_3_layers(self.block1_1, [64, 64, 256], "block1_2")
        self.block1_3 = self.res_block_3_layers(self.block1_2, [64, 64, 256], "block1_3")

        self.block2_1 = self.res_block_3_layers(self.block1_3, [128, 128, 512], "block2_1", True, 2)
        self.block2_2 = self.res_block_3_layers(self.block2_1, [128, 128, 512], "block2_2")
        self.block2_3 = self.res_block_3_layers(self.block2_2, [128, 128, 512], "block2_3")
        self.block2_4 = self.res_block_3_layers(self.block2_3, [128, 128, 512], "block2_4")

        self.block3_1 = self.res_block_3_layers(self.block2_4, [256, 256, 1024], "block3_1", True, 2)
        self.block3_2 = self.res_block_3_layers(self.block3_1, [256, 256, 1024], "block3_2")
        self.block3_3 = self.res_block_3_layers(self.block3_2, [256, 256, 1024], "block3_3")
        self.block3_4 = self.res_block_3_layers(self.block3_3, [256, 256, 1024], "block3_4")
        self.block3_5 = self.res_block_3_layers(self.block3_4, [256, 256, 1024], "block3_5")
        self.block3_6 = self.res_block_3_layers(self.block3_5, [256, 256, 1024], "block3_6")

        self.block4_1 = self.res_block_3_layers(self.block3_6, [512, 512, 2048], "block4_1", True, 2)
        self.block4_2 = self.res_block_3_layers(self.block4_1, [512, 512, 2048], "block4_2")
        self.block4_3 = self.res_block_3_layers(self.block4_2, [512, 512, 2048], "block4_3")

        self.pool2 = self.avg_pool(self.block4_3, 7, 1, "pool2")

        self.fc1 = self.fc_layer(self.pool2, 2048, label_num, "fc1200")

        if (last_layer_type == "sigmoid"):
            self.prob = tf.nn.sigmoid(self.fc1, name="prob")
        elif (last_layer_type == "softmax"):
            self.prob = tf.nn.softmax(self.fc1, name="prob")

        return self.pool2

    def res_block_3_layers(self, bottom, channel_list, name, change_dimension=False, block_stride=1):
        """
        bottom: input values (X)
        channel_list : number of channel in 3 layers
        name: block name
        """
        if (change_dimension):
            short_cut_conv = self.conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[2], block_stride,
                                             name + "_ShortcutConv")
            block_conv_input = self.batch_norm(short_cut_conv)
        else:
            block_conv_input = bottom

        block_conv_1 = self.conv_layer(bottom, 1, bottom.get_shape().as_list()[-1], channel_list[0], block_stride,
                                       name + "_lovalConv1")
        block_norm_1 = self.batch_norm(block_conv_1)
        block_relu_1 = tf.nn.relu(block_norm_1)

        block_conv_2 = self.conv_layer(block_relu_1, 3, channel_list[0], channel_list[1], 1, name + "_lovalConv2")
        block_norm_2 = self.batch_norm(block_conv_2)
        block_relu_2 = tf.nn.relu(block_norm_2)

        block_conv_3 = self.conv_layer(block_relu_2, 1, channel_list[1], channel_list[2], 1, name + "_lovalConv3")
        block_norm_3 = self.batch_norm(block_conv_3)
        block_res = tf.add(block_conv_input, block_norm_3)
        relu = tf.nn.relu(block_res)

        return relu

    def batch_norm(self, inputsTensor):
        """
        Batchnorm
        """
        _BATCH_NORM_DECAY = 0.99
        _BATCH_NORM_EPSILON = 1e-12
        return tf.layers.batch_normalization(inputs=inputsTensor, axis=3, momentum=_BATCH_NORM_DECAY,
                                             epsilon=_BATCH_NORM_EPSILON, center=True, scale=True,
                                             training=self.is_training)

    def avg_pool(self, bottom, kernal_size=2, stride=2, name="avg"):
        """
        bottom: input values (X)
        kernal_size : n * n kernal
        stride : stride
        name : block_layer name
        """
        print(name + ":")
        print(bottom.get_shape().as_list())
        return tf.nn.avg_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1],
                              padding='VALID', name=name)

    def max_pool(self, bottom, kernal_size=2, stride=2, name="max"):
        """
        bottom: input values (X)
        kernal_size : n * n kernal
        stride : stride
        name : block_layer name
        """
        print(name + ":")
        print(bottom.get_shape().as_list())
        return tf.nn.max_pool(bottom, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride, stride, 1],
                              padding='SAME', name=name)

    def conv_layer(self, bottom, kernal_size, in_channels, out_channels, stride, name):
        """
        bottom: input values (X)
        kernal_size : n * n kernal
        in_channels: number of input filters
        out_channels : number of output filters
        stride : stride
        name : block_layer name
        """
        print(name + ":")
        print(bottom.get_shape().as_list())
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(kernal_size, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)

            tf.summary.histogram('weight', filt)
            tf.summary.histogram('bias', conv_biases)

            return bias

    def fc_layer(self, bottom, in_size, out_size, name):
        """
        bottom: input values (X)
        in_size : number of input feature size
        out_size : number of output feature size
        """
        print(name + ":")
        print(bottom.get_shape().as_list())
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            tf.summary.histogram('weight', weights)
            tf.summary.histogram('bias', biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        """
        filter_size : 3 * 3
        in_channels : number of input filters
        out_channels : number of output filters
        name : block_layer name
        """
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0,
                                            stddev=1 / math.sqrt(float(filter_size * filter_size)))
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], 0.0, 1.0)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        """
        in_size : number of input feature size
        out_size : number of output feature size
        name : block_layer name
        """
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, stddev=1 / math.sqrt(float(in_size)))
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0, 1.0)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        """
        load variables from Loaded model or new generated random variables
        initial_value : random initialized value
        name: block_layer name
        index: 0,1 weight or bias
        var_name: name + "_filter"/"_bias"
        """
        if ((name, idx) in self.var_dict):
            print("Reuse Parameters...")
            print(self.var_dict[(name, idx)])
            return self.var_dict[(name, idx)]

        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var

    def save_npy(self, sess, npy_path="./model/Resnet-save.npy"):
        """
        Save this model into a npy file
        """
        assert isinstance(sess, tf.Session)

        self.data_dict = None
        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(("file saved", npy_path))
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count

>>>>>>> c7813c9... resnet
'''

block_sizes=[3, 4, 6, 3]指的是stage1(first pool)之后的4个layer的block数, 分别对应res2,res3,res4,res5,

    每一个layer的第一个block在shortcut上做conv+BN, 即Conv Block

inputs: (1, 720, 1280, 3)

initial_conv:

    conv2d_fixed_padding()

    1. kernel_size=7, 先做padding(1, 720, 1280, 3) -> (1, 726, 1286, 3)

    2. conv2d kernels=[7, 7, 3, 64], stride=2, VALID 卷积. 7x7的kernel, padding都为3, 为了保证左上角和卷积核中心点对其

       (1, 726, 1286, 3) -> (1, 360, 640, 64)

    3. BN, Relu (只有resnetv1在第一次conv后面做BN和Relu)

initial_max_pool:

    k=3, s=2, padding='SAME', (1, 360, 640, 64) -> (1, 180, 320, 64)

以下均为不使用bottleneck的building_block

block_layer1:

    (有3个block, layer间stride=1(上一层做pool了), 64个filter, 不使用bottleneck(若使用bottleneck 卷积核数量需乘4))

    1. 第一个block:

    Conv Block有projection_shortcut, 且strides可以等于1或者2

    Identity Block没有projection_shortcut, 且strides只能等于1

        `inputs = block_fn(inputs, filters, training, projection_shortcut, strides, data_format)`

        shortcut做[1, 1, 64, 64], stride=1的conv和BN, shape不变

        然后和主要分支里input做3次卷积后的结果相加, 一起Relu, 注意block里最后一次卷积后只有BN没有Relu

        input:    conv-bn-relu-conv-bn-relu-conv-bn  和shortcut相加后再做relu

        shortcut: conv-bn

        shortcut: [1, 1, 64, 64], s=1, (1, 180, 320, 64) -> (1, 180, 320, 64)

        input做两次[3, 3, 64, 64], s=1的卷积, shape不变(1, 180, 320, 64) -> (1, 180, 320, 64) -> (1, 180, 320, 64)

        inputs += shortcut, 再relu

    2. 对剩下的2个block, 每个block操作相同:

        `inputs = block_fn(inputs, filters, training, None, 1, data_format)`

        shortcut直接和input卷积结果相加, 不做conv-bn

        input做两次[3, 3, 64, 64], s=1的卷积, shape不变(1, 180, 320, 64) -> (1, 180, 320, 64) -> (1, 180, 320, 64)

        inputs += shortcut, 再relu

block_layer2/3/4同block_layer1, 只是每个layer的identity block数量不同, 卷积核数量和layer间stride也不同, 不过仍然只有第一个conv block的shortcut做conv-bn

block_layer2: 4个block, 128个filter, layer间stride=2 (因为上一层出来后没有pool)

    1. 第一个block:

        对shortcut做kernel=[1, 1, 64, 128], s=2的conv和BN, (1, 180, 320, 64) -> (1, 90, 160, 128)

        对主要分支先做kernel=[3, 3, 64, 128], s=2的卷积, padding='VALID', (1, 180, 320, 64) -> (1, 90, 160, 128)

                再做kernel=[3, 3, 128, 128], s=1的卷积, padding='SAME', (1, 90, 160, 128) -> (1, 90, 160, 128)

    2. 剩下的3个block, 每个block操作相同:

        shortcut不操作直接和结果相加做Relu

        对主要分支做两次[3, 3, 128, 128], s=1的卷积, padding='SAME', (1, 90, 160, 128) -> (1, 90, 160, 128) -> (1, 90, 160, 128)

block_layer3: 6个block, 256个filter, layer间stride=2

    1. 第一个block:

        对shortcut做kernel=[1, 1, 128, 256], s=2的conv和BN, (1, 90, 160, 128) -> (1, 45, 80, 256)

        对主要分支先做kernel=[3, 3, 128, 256], s=2的卷积, padding='VALID', (1, 90, 160, 128) -> (1, 45, 80, 256)

                再做kernel=[3, 3, 256, 256], s=1的卷积, padding='SAME', (1, 45, 80, 256) -> (1, 45, 80, 256)

    2. 剩下的5个block, 每个block操作相同:

        shortcut不操作直接和结果相加做Relu

        对主要分支做两次[3, 3, 256, 256], s=1的卷积, padding='SAME', (1, 45, 80, 256) -> (1, 45, 80, 256) -> (1, 45, 80, 256)

block_layer4: 3个block, 512个filter, layer间stride=2

    1. 第一个block:

        对shortcut做kernel=[1, 1, 256, 512], s=2的conv和BN, (1, 45, 80, 256) -> (1, 23, 40, 512)

        对主要分支先做kernel=[3, 3, 256, 512], s=2的卷积, padding='VALID', (1, 45, 80, 256) -> (1, 23, 40, 512)

                再做kernel=[3, 3, 512, 512], s=1的卷积, padding='SAME', (1, 23, 40, 512) -> (1, 23, 40, 512)

    2. 剩下的2个block, 每个block操作相同:

        shortcut不操作直接和结果相加做Relu

        对主要分支做两次[3, 3, 512, 512], s=1的卷积, padding='SAME', (1, 23, 40, 512) -> (1, 23, 40, 512)

avg_pool, 7*7

FC, output1000

softmax

输出prediction

'''
