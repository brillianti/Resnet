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
