import tensorflow as tf

from .layers import linear


class Network(object):
    def __init__(self, sess, name):
        self.sess = sess
        self.copy_op = None
        self.name = name
        self.var = {}

    def build_output_ops(self, input_layer, network_output_type,
                         output_size, weights_initializer,
                         biases_initializer, hidden_activation_fn,
                         output_activation_fn, trainable):
        self.outputs, self.var['w_out'], self.var['b_out'] = \
            linear(input_layer, output_size, weights_initializer,
                   biases_initializer, output_activation_fn, trainable, name='out')

        self.max_outputs = tf.reduce_max(self.outputs, reduction_indices=1)
        self.outputs_idx = tf.placeholder('int32', [None, None], 'outputs_idx')
        self.outputs_with_idx = tf.gather_nd(self.outputs, self.outputs_idx)
        self.actions = tf.argmax(self.outputs, axis=1)

    def run_copy(self):
        if self.copy_op is None:
            raise Exception("run `create_copy_op` first before copy")
        else:
            self.sess.run(self.copy_op)

    def create_copy_op(self, network):
        with tf.variable_scope(self.name):
            copy_ops = []

            for name in self.var.keys():
                copy_op = self.var[name].assign(network.var[name])
                copy_ops.append(copy_op)

            self.copy_op = tf.group(*copy_ops, name='copy_op')

    def calc_actions(self, observation):
        return self.actions.eval({self.inputs: observation}, session=self.sess)

    def calc_outputs(self, observation):
        return self.outputs.eval({self.inputs: observation}, session=self.sess)

    def calc_max_outputs(self, observation):
        return self.max_outputs.eval({self.inputs: observation}, session=self.sess)

    def calc_outputs_with_idx(self, observation, idx):
        return self.outputs_with_idx.eval(
            {self.inputs: observation, self.outputs_idx: idx}, session=self.sess)
