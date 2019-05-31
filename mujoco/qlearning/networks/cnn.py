import os
import tensorflow as tf

from .layers import conv2d, linear, initializers
from .network import Network


class CNN(Network):
    def __init__(self, sess,
                 data_format,
                 observation_dims,
                 trainable=True,
                 hidden_activation_fn=tf.nn.relu,
                 output_activation_fn=None,
                 weights_initializer=initializers.xavier_initializer(),
                 biases_initializer=tf.constant_initializer(0.1),
                 network_output_type='dueling',
                 name='CNN',
                 successor=False):
        super(CNN, self).__init__(sess, name)

        if data_format == 'NHWC':
            self.inputs = tf.placeholder(
                'float32',
                [None] + observation_dims,
                name='inputs')
        elif data_format == 'NCHW':
            self.inputs = tf.placeholder(
                'float32',
                [None] + observation_dims,
                name='inputs')
        else:
            raise ValueError("unknown data_format : %s" % data_format)

        with tf.variable_scope(name):
            self.l1, self.var['l1_w'], self.var['l1_b'] = linear(
                self.inputs, 32,
                weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l1_conv')

            self.l2, self.var['l2_w'], self.var['l2_b'] = linear(
                self.l1, 32,
                weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l2_conv')

            self.phiVal, self.var['l3_w'], self.var['l3_b'] = linear(
                self.l2, 8,
                weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l3_conv')

            self.l4, self.var['l4_w'], self.var['l4_b'] = linear(
                self.phiVal, 8,
                weights_initializer, biases_initializer,
                hidden_activation_fn, data_format, name='l4_conv')

            layer = self.l4
            output_size = 8

            self.build_output_ops(
                layer, network_output_type,
                output_size, weights_initializer,
                biases_initializer, hidden_activation_fn,
                output_activation_fn, trainable)

    def calc_phiVal(self, observation):
        return self.phiVal.eval({self.inputs: observation}, session=self.sess)
