# -*- coding:utf-8 -*-

import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_LeNet5_inference
import mnist_LeNet5_train

# 定义调用最新模型评估模型准确率的时间间隔
EVAL_INTERVAL_SECS = 5

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            mnist.validation.num_examples, 
            mnist_LeNet5_inference.IMAGE_SIZE, 
            mnist_LeNet5_inference.IMAGE_SIZE, 
            mnist_LeNet5_inference.NUM_CHANNELS],
            name = "x-input")
        y_ = tf.placeholder(tf.float32, [
            mnist.validation.num_examples, 
            mnist_LeNet5_inference.OUTPUT_NODES],
            name = "y-input")
        
        xs = np.reshape(mnist.validation.images, (
                    mnist.validation.num_examples, 
                    mnist_LeNet5_inference.IMAGE_SIZE, 
                    mnist_LeNet5_inference.IMAGE_SIZE,
                    mnist_LeNet5_inference.NUM_CHANNELS)
                    )
        y = mnist_LeNet5_inference.inference(xs, False, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_average = tf.train.ExponentialMovingAverage(
            mnist_LeNet5_train.MOVING_AVERAGE_RATE)
        variables_to_restore = variable_average.variables_to_restore()

        saver = tf.train.Saver(variables_to_restore)
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(
                    mnist_LeNet5_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split(
                        "/")[-1].split("/")[-1]
                    accuracy_score = sess.run(
                        accuracy, feed_dict={x:xs, y_: mnist.validation.labels})
                    print("After %s train step(s), validation accuracy=%g" % (
                        global_step, accuracy_score))

def main(argv=None):
    mnist = input_data.read_data_sets('../../Data', one_hot=True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()