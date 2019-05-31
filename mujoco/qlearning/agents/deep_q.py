import os
import time
import numpy as np
import tensorflow as tf
import pickle
from logging import getLogger

from .agent import Agent

logger = getLogger(__name__)


class DeepQ(Agent):
    def __init__(self, sess, pred_network, env, testEnv,
                 conf, target_network=None):
        super(DeepQ, self).__init__(sess, pred_network, env, testEnv,
                                    conf, target_network=target_network)

        # Optimizer
        with tf.variable_scope('optimizer'):
            self.targets = tf.placeholder('float32', [None], name='target_q_t')
            self.actions = tf.placeholder('int64', [None], name='action')

            actions_one_hot = tf.one_hot(
                self.actions, self.env.action_size, 1.0, 0.0, name='action_one_hot')
            pred_q = tf.reduce_sum(
                self.pred_network.outputs * actions_one_hot, reduction_indices=1, name='q_acted')

            self.delta = self.targets - pred_q
            self.clipped_error = tf.where(tf.abs(self.delta) < 1.0,
                                          0.5 * tf.square(self.delta),
                                          tf.abs(self.delta) - 0.5, name='clipped_error')

            self.loss = tf.reduce_mean(self.clipped_error, name='loss')

            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.learning_rate,
                                                   self.t_op,
                                                   self.learning_rate_decay_step,
                                                   self.learning_rate_decay,
                                                   staircase=True))

            optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01)

            #  if self.max_grad_norm is not None:
            grads_and_vars = optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(
                        grad, self.max_grad_norm), var)
            self.optim = optimizer.apply_gradients(grads_and_vars)

    def observe(self, observation, reward, action, terminal):
        reward = max(self.min_r, min(self.max_r, reward))

        self.history.add(observation)
        self.experience.add(observation, reward, action, terminal)

        # q, loss, is_update
        result = [], 0, False

        if self.t > self.t_learn_start:
            if self.t % self.t_train_freq == 0:
                result = self.q_learning_minibatch()

            if self.t % self.t_target_q_update_freq == self.t_target_q_update_freq - 1:
                self.update_target_q_network()

        return result

    def q_learning_minibatch(self):
        if self.experience.count < 4:
            return [], 0, False
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.experience.sample()

        terminal = np.array(terminal) + 0.

        if self.double_q:
            # Double Q-learning
            pred_action = self.pred_network.calc_actions(s_t_plus_1)
            q_t_plus_1_with_pred_action = self.target_network.calc_outputs_with_idx(
                s_t_plus_1, [[idx, pred_a] for idx, pred_a in enumerate(pred_action)])
            target_q_t = (1. - terminal) * self.discount_r * \
                q_t_plus_1_with_pred_action + reward
        else:
            # Deep Q-learning
            max_q_t_plus_1 = self.target_network.calc_max_outputs(s_t_plus_1)
            target_q_t = (1. - terminal) * self.discount_r * \
                max_q_t_plus_1 + reward

        _, q_t, loss = self.sess.run([self.optim, self.pred_network.outputs, self.loss], {
            self.targets: target_q_t,
            self.actions: action,
            self.pred_network.inputs: s_t,
        })

        return q_t, loss, True


class DeepSuccessor(Agent):
    def __init__(self, sess, pred_network, env, testEnv,
                 conf, target_network=None):
        super(DeepSuccessor, self).__init__(sess, pred_network, env, testEnv,
                                            conf, target_network=target_network)

        # Optimizer
        with tf.variable_scope('optimizer'):
            self.targets = tf.placeholder(
                'float32', [None, 8], name='target_sr_t')

            self.delta = tf.reduce_mean(
                tf.square(self.targets - self.pred_network.outputs),
                axis=1)

            self.clipped_error = tf.where(self.delta < 1.0,
                                          self.delta,
                                          tf.sqrt(self.delta) - 0.5, name='clipped_error')

            self.loss = tf.reduce_mean(self.clipped_error, name='loss')

            self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                                               tf.train.exponential_decay(
                                                   self.learning_rate,
                                                   self.t_op,
                                                   self.learning_rate_decay_step,
                                                   self.learning_rate_decay,
                                                   staircase=True))

            optimizer = tf.train.RMSPropOptimizer(
                self.learning_rate_op, momentum=0.95, epsilon=0.01)

            if self.max_grad_norm is not None:
                grads_and_vars = optimizer.compute_gradients(self.loss)
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grads_and_vars[idx] = (tf.clip_by_norm(
                            grad, self.max_grad_norm), var)
                self.optim = optimizer.apply_gradients(grads_and_vars)
            else:
                self.optim = optimizer.minimize(self.loss)

    def miniobserve(self, observation, reward, action, terminal):
        self.history.add(observation)

    def observe(self, observation, reward, action, terminal):
        reward = max(self.min_r, min(self.max_r, reward))

        self.history.add(observation)
        self.experience.add(observation, reward, action, terminal)

        # q, loss, is_update
        result = [], 0, False

        if self.t > self.t_learn_start:
            if self.t % self.t_train_freq == 0:
                result = self.q_learning_minibatch()

            if self.t % self.t_target_q_update_freq == self.t_target_q_update_freq - 1:
                self.update_target_q_network()

        return result

    def q_learning_minibatch(self):
        if self.experience.count < 4:
            return [], 0, False
        else:
            s_t, action, reward, s_t_plus_1, terminal = self.experience.sample()

        terminal = np.array(terminal) + 0.

        if self.double_q:
            pass

        srOutputs = self.target_network.calc_outputs(s_t_plus_1)
        phiVal = self.target_network.calc_phiVal(s_t_plus_1)
        terminal = np.expand_dims(terminal, 1)

        target_sr = (1. - terminal) * self.discount_r * srOutputs + phiVal

        _, srVal, loss = self.sess.run([self.optim, self.pred_network.outputs, self.loss], {
            self.targets: target_sr,
            self.pred_network.inputs: s_t,
        })

        return srVal, loss, True

    def train(self, t_max):
        tf.global_variables_initializer().run()

        start_t = self.t_op.eval(session=self.sess)
        observation, reward, terminal = self.new_game()

        self.history.add(observation)

        startTime = time.time()

        STEPLOGSIZE = 10000
        for self.t in range(start_t, t_max):
            ep = 1

            self.t_add_op.eval(session=self.sess)

            if self.t % STEPLOGSIZE == 0:
                diff = time.time() - startTime
                logger.info("At Step " + str(self.t) +
                            " : " + str("%.4f" % (float(STEPLOGSIZE) / (diff))) +
                            " steps/sec")
                startTime = time.time()

            if self.t % 50000 == 0 or self.t == t_max - 1:

                logger.info("SAVING MODEL")
                self.saver.save(self.sess, "./checkpoints/" +
                                str(self.t) + "_model.ckpt")
                logger.info("MODEL SAVED")

            # 1. predict
            action = self.predict(self.history.get(), ep)
            # 2. act
            observation, reward, terminal, info = self.env.step(
                action, is_training=True)
            # 3. observe
            sr, loss, is_update = self.observe(
                observation, reward, action, terminal)

            if terminal:
                observation, reward, terminal = self.new_game()

    def restoreCkpt(self, restorePath):
        tf.global_variables_initializer().run()
        self.saver.restore(self.sess, restorePath)


