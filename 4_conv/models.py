#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
import argparse
import os
import time


class CNNModel(object):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def _define_graph(self, input_tensor, train, regularizer):
        with tf.variable_scope('layer1-conv1'):
            conv1_weights = tf.get_variable(
                "weight", [self.args.conv1_size, self.args.conv1_size,
                           self.args.num_channels, self.args.conv1_deep],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv1_biases = tf.get_variable(
                "bias", [self.args.conv1_deep], initializer=tf.constant_initializer(0.0))
            conv1 = tf.nn.conv2d(input_tensor, conv1_weights,
                                 strides=[1, 1, 1, 1], padding='SAME')
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

        with tf.name_scope("layer2-pool1"):
            pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[
                                   1, 2, 2, 1], padding="SAME")

        with tf.variable_scope("layer3-conv2"):
            conv2_weights = tf.get_variable(
                "weight", [self.args.conv2_size, self.args.conv2_size,
                           self.args.conv1_deep, self.args.conv2_deep],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
            conv2_biases = tf.get_variable(
                "bias", [self.args.conv2_deep], initializer=tf.constant_initializer(0.0))
            conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[
                                 1, 1, 1, 1], padding='SAME')
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

        with tf.name_scope("layer4-pool2"):
            pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[
                                   1, 2, 2, 1], padding='SAME')
            pool_shape = pool2.get_shape().as_list()
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(pool2, [-1, nodes])

        # 全连接层_1
        with tf.variable_scope('layer5-fc1'):
            fc1_weights = tf.get_variable("weight", [nodes, self.args.fc1_size],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None:
                tf.add_to_collection('losses', regularizer(fc1_weights))
            fc1_biases = tf.get_variable(
                "bias", [self.args.fc1_size], initializer=tf.constant_initializer(0.1))

            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
            if train:
                fc1 = tf.nn.dropout(fc1, 0.5)  # 一般在全连接层使用dropout

        # 全连接层_2
        with tf.variable_scope('layer6-fc2'):
            fc2_weights = tf.get_variable("weight", [self.args.fc2_size, self.args.num_labels],
                                          initializer=tf.truncated_normal_initializer(stddev=0.1))
            if regularizer != None:
                tf.add_to_collection('losses', regularizer(fc2_weights))
            fc2_biases = tf.get_variable(
                "bias", [self.args.num_labels], initializer=tf.constant_initializer(0.1))
            logit = tf.matmul(fc1, fc2_weights) + fc2_biases

        return logit

    def fit(self):
        x = tf.placeholder(tf.float32, [
            None,
            self.args.image_size,
            self.args.image_size,
            self.args.num_channels],
            name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, self.args.output_node], name='y-input')

        regularizer = tf.contrib.layers.l2_regularizer(
            self.args.regulation_rate)
        y = self._define_graph(x, True, regularizer)
        global_step = tf.Variable(0, trainable=False)

        # 定义损失函数、学习率、滑动平均操作以及训练过程。
        variable_averages = tf.train.ExponentialMovingAverage(
            self.args.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(
            tf.trainable_variables())
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        learning_rate = tf.train.exponential_decay(
            self.args.learning_rate_base,
            global_step,
            self.data.train_dataset.shape[0] /
            self.args.batch_size,  self.args.learning_rate_decay,
            staircase=True)

        train_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # 初始化TensorFlow持久化类。
        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for epoch in range(self.args.training_steps):
                xs, ys = self.data.next_batchs(self.args.batch_size)
                _, loss_value, step = sess.run(
                    [train_op, loss, global_step], feed_dict={x: xs, y_: ys})

                if epoch % 100 == 0:
                    print("After %d training step(s), loss on training batch is %g." % (
                        step, loss_value))
                    saver.save(sess, os.path.join(self.args.model_save_path,
                                                  self.args.model_name), global_step=global_step)

    def test(self):
        with tf.Graph().as_default() as g:
            x = tf.placeholder(tf.float32, [
                None,
                self.args.image_size,
                self.args.image_size,
                self.args.num_channels],
                name='x-input')
            y_ = tf.placeholder(
                tf.float32, [None, self.args.output_node], name='y-input')

            validate_feed = {x: self.data.valid_dataset,
                             y_: self.data.valid_labels}

            y = self._define_graph(x, None, None)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            variable_averages = tf.train.ExponentialMovingAverage(
                self.args.moving_average_decay)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            while True:
                with tf.Session() as sess:
                    ckpt = tf.train.get_checkpoint_state(self.args.model_save_path)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        global_step = ckpt.model_checkpoint_path.split(
                            '/')[-1].split('-')[-1]
                        accuracy_score = sess.run(
                            accuracy, feed_dict=validate_feed)
                        print("After %s training step(s), validation accuracy = %g" % (
                            global_step, accuracy_score))
                    else:
                        print('No checkpoint file found')
                        return
                time.sleep(self.args.eval_interval_secs)
