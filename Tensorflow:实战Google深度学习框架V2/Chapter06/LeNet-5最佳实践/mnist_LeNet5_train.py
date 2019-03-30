# -*- coding: utf-8 -*-

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

import mnist_LeNet5_inference 

# 神经网络参数设置
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DACAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_RATE = 0.99

# 模型保存路径
MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "mnist_LeNet5.ckpt"

def train(mnist):
    # 定义输入输出
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE, 
        mnist_LeNet5_inference.IMAGE_SIZE, 
        mnist_LeNet5_inference.IMAGE_SIZE,
        mnist_LeNet5_inference.NUM_CHANNELS],
        name="x-input")
    y_ = tf.placeholder(
        tf.float32, [None, mnist_LeNet5_inference.NUM_LABELS], 
        name="y-input")
    # 定义正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 使用前向传播方法
    y = mnist_LeNet5_inference.inference(x, True, regularizer)
    # 定义滑动平均值
    global_step = tf.Variable(0, trainable = False)
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_RATE, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 定义交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 定义损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection("losses")) 
    # 定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, 
        global_step, 
        mnist.train.num_examples / BATCH_SIZE, 
        LEARNING_RATE_DACAY)
    # 定义训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step = global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")
    
    # 初始化模型保存类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE, 
                mnist_LeNet5_inference.IMAGE_SIZE, 
                mnist_LeNet5_inference.IMAGE_SIZE,
                mnist_LeNet5_inference.NUM_CHANNELS))

            _, loss_values, steps = sess.run([train_op, loss, global_step], \
                feed_dict={x: reshaped_xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s), loss in train batch is %g" % (steps, loss_values))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), 
                    global_step=global_step)        

def main(argv=None):
    mnist = input_data.read_data_sets("../../Data", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()