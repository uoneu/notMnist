#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
import time

import input_data as inputs
import nn
import train


# 加载的时间间隔。
EVAL_INTERVAL_SECS = 10

def evaluate():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [
            None,
            nn.IMAGE_SIZE,
            nn.IMAGE_SIZE,
            nn.NUM_CHANNELS],
            name='x-input')
        y_ = tf.placeholder(tf.float32, [None, nn.OUTPUT_NODE], name='y-input')
        validate_feed = {x: inputs.valid_dataset, y_: inputs.valid_labels }

        y = nn.inference(x, None, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)



def main(argv=None):
    evaluate()


if __name__ == '__main__':
    main()