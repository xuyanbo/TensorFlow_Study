# -*- coding:utf-8 -*-

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

# 定义调用最新模型评估模型准确率的间隔时间
EVAL_INTERVAL_SECS = 5

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出
        x = tf.placeholder(
            tf.float32, [None, mnist_inference.INPUT_NODES], name="x-input")
        y_ = tf.placeholder(
            tf.float32, [None, mnist_inference.OUTPUT_NODES], name="y-input")
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = mnist_inference.inference(x, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
        variable_averages = tf.train.ExponentialMovingAverage(
            mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_resore = variable_averages.variables_to_restore()
        
        saver = tf.train.Saver(variable_to_resore)
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODLE_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split(
                        "/")[-1].split("/")[-1]
                    accuracy_score = sess.run(accuracy, feed_dict = validate_feed)
                    print("After %s training step(s), validation accuracy=%g"%(
                        global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
            time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist = input_data.read_data_sets('../../Data', one_hot=True)
    evaluate(mnist)

if __name__ == "__main__":
    tf.app.run()