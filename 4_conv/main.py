#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from six.moves import cPickle as pickle

import numpy as np
import argparse

import models
import data_utils


parser = argparse.ArgumentParser()
arg = parser.add_argument

arg('--data-path', default="../notMNIST.pickle")

arg('--image-size', type=int, default=28)
arg('--num-channels', type=int, default=1)
arg('--num-labels', type=int, default=10)

arg('--input-node', type=int, default=784)
arg('--output-node', type=int, default=10)
arg('--conv1-deep', type=int, default=32)
arg('--conv1-size', type=int, default=5)
arg('--conv2-deep', type=int, default=64)
arg('--conv2-size', type=int, default=5)
arg('--fc1-size', type=int, default=512)
arg('--fc2-size', type=int, default=512)


arg('--batch-size', type=int, default=100)
arg('--learning-rate-base', type=float, default=0.01)
arg('--learning_rate_decay', type=float, default=0.99)
arg('--regulation-rate', type=float, default=0.0001)
arg('--moving-average-decay', type=float, default=0.99)
arg('--training-steps', type=int, default=10000)

arg('--model-save-path', default="./models/")
arg('--model-name', default="model")

arg('--eval-interval-secs', type=int, default=10)

args = parser.parse_args()


def train(data):
    model = models.CNNModel(data, args)
    model.fit()


def test(data):
    model = models.CNNModel(data, args)
    model.test()


if __name__ == '__main__':
    data = data_utils.DataLoader(args)
    #train(data)
    test(data)
