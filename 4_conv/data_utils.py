#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from six.moves import cPickle as pickle

import numpy as np
import argparse


class DataLoader(object):
    def __init__(self, args):
        self.args = args
        self.load_data()
        self._index_in_epoch = 0  # 在一个epoch中的index

    def reformat(self, dataset, labels):
        dataset = dataset.reshape(
            (-1, self.args.image_size, self.args.image_size, self.args.num_channels)).astype(np.float32)
        labels = (np.arange(self.args.num_labels) ==
                  labels[:, None]).astype(np.float32)
        return dataset, labels

    def load_data(self):
        with open(self.args.data_path, 'rb') as f:
            save = pickle.load(f)
            self.train_dataset = save['train_dataset']
            self.train_labels = save['train_labels']
            self.valid_dataset = save['valid_dataset']
            self.valid_labels = save['valid_labels']
            self.test_dataset = save['test_dataset']
            self.test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            print('Training set', self.train_dataset.shape,
                  self.train_labels.shape)
            print('Validation set', self.valid_dataset.shape,
                  self.valid_labels.shape)
            print('Test set', self.test_dataset.shape, self.test_labels.shape)
        self.train_dataset, self.train_labels = self.reformat(
            self.train_dataset, self.train_labels)
        self.valid_dataset, self.valid_labels = self.reformat(
            self.valid_dataset, self.valid_labels)
        self.test_dataset, self.test_labels = self.reformat(
            self.test_dataset, self.test_labels)
        print('Training set', self.train_dataset.shape, self.train_labels.shape)
        print('Validation set', self.valid_dataset.shape, self.valid_labels.shape)
        print('Test set', self.test_dataset.shape, self.test_labels.shape)

    def next_batchs(self, batch_size=128):
        start = self._index_in_epoch
        num_examples = self.train_dataset.shape[0]

        if start + batch_size > num_examples:  # epoch的结尾和下一个epoch的开头
            # Get the rest examples in this epoch
            rest_num_examples = num_examples - start  # 最后不够一个batch还剩下几个
            data_rest_part = self.train_dataset[start:num_examples]
            labels_rest_part = self.train_labels[start:num_examples]
            # Shuffle the data
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self.train_dataset[start:end]
            labels_new_part = self.train_labels[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self.train_dataset[start:end], self.train_labels[start:end]


if __name__ == '__main__':
    dl = DataLoader('../notMNIST.pickle', args)
    for i in range(1000):
        x,_ = dl.next_batchs(10021)
        print(x.shape)