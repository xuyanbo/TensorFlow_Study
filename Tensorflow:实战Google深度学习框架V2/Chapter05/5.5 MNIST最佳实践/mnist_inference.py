# -*- coding : utf-8 -*-
import tensorflow as tf

INPUT_NODES = 784
OUTPUT_NODES = 10
LAYER1_NODES = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        "weights", shape, 
        initializer=tf.truncated_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection("losses", regularizer(weights))

    return weights

def inference(input_tensor, regularizer):
    with tf.variable_scope("layer1"):
        weights = get_weight_variable(
            [INPUT_NODES, LAYER1_NODES], regularizer)
        bias = tf.get_variable(
            "bias", [LAYER1_NODES],
            initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + bias)
    
    with tf.variable_scope("layer2"):
        weights = get_weight_variable(
            [LAYER1_NODES, OUTPUT_NODES], regularizer)
        bias = tf.get_variable(
            "bias", [OUTPUT_NODES], 
            initializer=tf.constant_initializer(0.0))
        # 最后一层不使用激活函数！！！！！important
        layer2 = tf.matmul(layer1, weights) + bias

    return layer2