#!/bin/bash python3
import random
import tensorflow as tf

from networks.cnn import CNN
from networks.cnn2 import CNN2
from env import FetchEnvironment
#  from agents.deep_q import DeepQ
from agents.deep_q import DeepSuccessor

flags = tf.app.flags

# GPU
flags.DEFINE_boolean(
    'use_gpu', False,
    'Whether to use gpu or not. gpu use NHWC and gpu use NCHW for data_format')

#  DQN
flags.DEFINE_boolean(
    'is_train', True,
    'Whether to do training or testing')
flags.DEFINE_boolean(
    'double_q', True,
    'Whether to use double Q-learning')
flags.DEFINE_string(
    'network_output_type', 'normal',
    'The type of network output [normal, dueling]')

# Environment
flags.DEFINE_boolean(
    'deepmindLab', False,
    'The environment type')

flags.DEFINE_boolean(
    'vizDoom', False,
    'The environment type')

flags.DEFINE_boolean(
    'fetch', True,
    'The environment type')

flags.DEFINE_string(
    'observation_dims', '[10]',
    'The dimension of gym observation')

# Training
flags.DEFINE_integer(
    'max_delta', None,
    'The maximum value of delta')
flags.DEFINE_integer(
    'min_delta', None,
    'The minimum value of delta')
flags.DEFINE_float(
    'ep_start', 1.,
    'The value of epsilon at start in e-greedy')
flags.DEFINE_float(
    'ep_end', 0.01,
    'The value of epsilnon at the end in e-greedy')
flags.DEFINE_integer(
    'batch_size', 32,
    'The size of batch for minibatch training')
flags.DEFINE_integer(
    'max_grad_norm', None,
    'The maximum norm of gradient while updating')
flags.DEFINE_float('discount_r', 0.99, 'The discount factor for reward')

# Timer
flags.DEFINE_integer(
    't_train_freq', 4, '')

flags.DEFINE_boolean(
    'successor', True,
    'Collect SR')

# Below numbers will be multiplied by scale
flags.DEFINE_integer(
    'scale', 1000,
    'The scale for big numbers')
flags.DEFINE_integer(
    'memory_size', 20,
    'The size of experience memory (*= scale)')
flags.DEFINE_integer(
    't_target_q_update_freq', 1,
    'The frequency of target network to be updated (*= scale)')
flags.DEFINE_integer(
    't_test', 1,
    'The maximum number of t while training (*= scale)')
flags.DEFINE_integer(
    't_ep_end', 100,
    'The time when epsilon reach ep_end (*= scale)')
flags.DEFINE_integer(
    't_train_max', 300,
    'The maximum number of t while training (*= scale)')
flags.DEFINE_float(
    't_learn_start', 5,
    'The time when to begin training (*= scale)')
flags.DEFINE_float(
    'learning_rate_decay_step', 5,
    'The learning rate of training (*= scale)')
flags.DEFINE_integer(
    'max_r', +10,
    'The maximum value of clipped reward')
flags.DEFINE_integer(
    'min_r', -10,
    'The minimum value of clipped reward')


# Optimizer
flags.DEFINE_float(
    'learning_rate', 0.00025,
    'The learning rate of training')
flags.DEFINE_float(
    'learning_rate_minimum', 0.00025,
    'The minimum learning rate of training')
flags.DEFINE_float(
    'learning_rate_decay', 0.96,
    'The decay of learning rate of training')
flags.DEFINE_float(
    'decay', 0.99,
    'Decay of RMSProp optimizer')
flags.DEFINE_float(
    'momentum', 0.0,
    'Momentum of RMSProp optimizer')
flags.DEFINE_float(
    'gamma', 0.99,
    'Discount factor of return')
flags.DEFINE_float(
    'beta', 0.01,
    'Beta of RMSProp optimizer')

# Debug
flags.DEFINE_boolean(
    'display', False,
    'Whether to do display the game screen or not')
flags.DEFINE_integer(
    'random_seed', 123,
    'Value of random seed')
flags.DEFINE_string(
    'tag', '',
    'The name of tag for a model, only for debugging')
flags.DEFINE_boolean(
    'allow_soft_placement', True,
    'Whether to use part or all of a GPU')

# Internal
# It is forbidden to set a flag that is not defined
flags.DEFINE_string(
    'data_format', 'NCHW',
    'INTERNAL USED ONLY')

conf = flags.FLAGS

# set random seed
tf.set_random_seed(conf.random_seed)
random.seed(conf.random_seed)


def main(_):
    # preprocess
    conf.observation_dims = eval(conf.observation_dims)

    if conf.use_gpu:
        conf.data_format = 'NCHW'
    else:
        conf.data_format = 'NHWC'

    #  Allow soft placement to occupy as much GPU as needed
    sess_config = tf.ConfigProto(
        log_device_placement=False, allow_soft_placement=True)
    sess_config.gpu_options.allow_growth = True

    with tf.Session(config=sess_config) as sess:
        env = FetchEnvironment(conf.observation_dims)
        testEnv = FetchEnvironment(conf.observation_dims)

        pred_network = CNN(sess=sess,
                           data_format=conf.data_format,
                           observation_dims=conf.observation_dims,
                           name='pred_network', trainable=True,
                           successor=conf.successor)
        target_network = CNN(sess=sess,
                             data_format=conf.data_format,
                             observation_dims=conf.observation_dims,
                             name='target_network', trainable=False,
                             successor=conf.successor)

        if conf.successor:
            agent = DeepSuccessor(sess, pred_network, env, testEnv,
                                  conf, target_network=target_network)

            agent.train(conf.t_train_max)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    tf.app.run()
