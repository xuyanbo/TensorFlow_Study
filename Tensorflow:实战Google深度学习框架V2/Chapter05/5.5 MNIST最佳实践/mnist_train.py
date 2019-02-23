# -*- coding: utf-8 -*-

import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference

# 定义神经网络参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存路径
MODLE_SAVE_PATH = './model/'
MODLE_NAME = 'model.ckpt'

def train(mnist):
    # 定义输入输出
    x = tf.placeholder(
        tf.float32, [None, mnist_inference.INPUT_NODES], name='x-input')
    y_ = tf.placeholder(
        tf.float32, [None, mnist_inference.OUTPUT_NODES], name='y-input')
    
    # 定义正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 使用mnist_inference定义前向传播
    y = mnist_inference.inference(x, regularizer)
    global_step = tf.Variable(0, trainable = False)
    # 定义滑动平均值
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())
    # 定义交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 定义损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    # 定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, 
        global_step, 
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)
    # 训练过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
        .minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name="train")

    # 初始TensorFlow的持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], 
            feed_dict={x:xs, y_:ys})
            # 每隔1000次保存一次模型
            if i % 1000 == 0:
                print('After %d training step(s), loss on training batch is %g.' % (step, loss_value))
                saver.save(
                    sess, os.path.join(MODLE_SAVE_PATH, MODLE_NAME),
                    global_step=global_step)

def main(argv=None):
    mnist = input_data.read_data_sets("../../Data/", one_hot=True)
    train(mnist)

if __name__ == "__main__":
    tf.app.run()