# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 定义网络结构参数
INPUT_NODES = 784
OUTPUT_NODES = 10

# 图片参数
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积层深度、尺寸和步长
CONV1_DEPT = 32
CONV1_SIZE = 5
CONV1_STRIDE = 1

# 第一层池化层尺寸和步长
POOL1_SIZE = 2
POOL1_STRIDE = 2

# 第三层卷积层深度、尺寸和步长
CONV2_DEPT = 64
CONV2_SIZE = 5
CONV2_STRIDE = 1

# 第四层池化层尺寸和步长
POOL2_SIZE = 2
POOL2_STRIDE = 2

# 第五层全连接层节点数
FC_SIZE = 512

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape, 
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    
    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))
    
    return weights

def inference(input_tensor, train, regularizer):
    # 定义第一层：卷积层 input: 28*28*1 output: 28*28*32
    with tf.variable_scope("layer1-conv1"):
        weights = get_weight_variable(
            [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEPT], None)
        conv1_biases = tf.get_variable(
            "biases", [CONV1_DEPT], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(
            input_tensor, weights, [1, CONV1_STRIDE, CONV1_STRIDE, 1], 
            padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 定义第二层：池化层 input: 28*28*32 output: 14*14*32
    with tf.variable_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(
            relu1, [1, POOL1_SIZE, POOL1_SIZE, 1], 
            [1, POOL1_STRIDE, POOL1_STRIDE, 1], padding='SAME')
    
    # 定义第三层：卷积层 input: 14*14*32 output: 14*14*64
    with tf.variable_scope("layer3-conv2"):
        weights = get_weight_variable(
            [CONV2_SIZE, CONV2_SIZE, CONV1_DEPT, CONV2_DEPT], None)
        conv2_biases = tf.get_variable(
            "biases", [CONV2_DEPT], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(
            pool1, weights, [1, CONV2_STRIDE, CONV2_STRIDE, 1], 
            padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    
    # 定义第四层：池化层 input：14*14*64 output：7*7*64
    with tf.variable_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(
            relu2, [1, POOL2_SIZE, POOL2_SIZE, 1], 
            [1, POOL2_STRIDE, POOL2_STRIDE, 1], padding='SAME')
        
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 定义第五层：全连接层
    # 池化层输出转化为全连接格式输入
    with tf.variable_scope("layer5-fc1"):
        fc1_weights = get_weight_variable([nodes, FC_SIZE], regularizer)
        # 不能设置为常数0，否则结果不对！！！important
        fc1_biases = tf.get_variable()
            "fc1_biased", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    
    with tf.Session() as sess:
            print(sess.run(tf.shape(fc1)))
    with tf.variable_scope("layer6-fc2"):
        fc2_weights = get_weight_variable([FC_SIZE, OUTPUT_NODES], regularizer)
        fc2_biases = tf.get_variable(
            "bias", [OUTPUT_NODES], initializer=tf.constant_initializer(0.1))

        fc2 = tf.matmul(fc1, fc2_weights) + fc2_biases
    
    return fc2