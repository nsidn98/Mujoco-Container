from __future__ import print_function

import random
import numpy as np
import cv2
import pickle

import gym
import copy

imresize = cv2.resize
imwrite = cv2.imwrite


class Environment(object):
    def __init__(self, env_name, n_action_repeat, max_random_start,
                 observation_dims, data_format, display, use_cumulated_reward=False):
        self.env = None

        self.n_action_repeat = n_action_repeat
        self.max_random_start = max_random_start
        self.action_size = self.env.action_space.n

        self.display = display
        self.data_format = data_format
        self.observation_dims = observation_dims
        self.use_cumulated_reward = use_cumulated_reward

    def new_game(self):
        return self.preprocess(self.env.reset()), 0, False

    def new_random_game(self):
        return self.new_game()

    def step(self, action, is_training=False):
        observation, reward, terminal, info = self.env.step(action)
        if self.display:
            self.env.render()
        return self.preprocess(observation), reward, terminal, info

    def preprocess(self):
        raise NotImplementedError()


class FetchEnvironment(Environment):
    def __init__(self, observation_dims, maxEpLen=30000, pseudo=False):

        self.env = gym.make('FetchReach-v1')

        ans = self.env.reset()
        self.obs = ans["observation"]
        self.numSteps = 0
        self.maxEpLen = maxEpLen
        self.prevPhi = None

        self.actionSet = [
            np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.intc),    # Forward
            np.array([0, 0, 0, -1, 0, 0, 0], dtype=np.intc),   # Backward
            np.array([90, 0, 0, 0, 0, 0, 0], dtype=np.intc),   # Right rotate
            np.array([-90, 0, 0, 0, 0, 0, 0], dtype=np.intc)]  # Left Rotate

        self.observation_dims = observation_dims
        self.action_size = len(self.actionSet)

    def new_game(self):
        ans = self.env.reset()
        self.numSteps = 0
        stateImg = ans["observation"]
        self.prevPhi = None

        return stateImg, 0, False

    def new_random_game(self):
        return self.new_game()

    def termChecker(self):
        if self.numSteps > self.maxEpLen:
            return True
        elif self.obs[0] <= 1.00 or self.obs[0] >= 1.510:
            return True
        elif self.obs[1] <= 0.38 and self.obs[1] >= 1.1:
            return True
        elif self.obs[2] >= 0.875:
            return True
        else:
            return False

    def step(self, action, is_training=False, phi=None):

        if self.numSteps == 1 or self.numSteps % 500 == 0:
            self.u1 = np.random.uniform(0, 1, 3)
            self.u2 = np.random.uniform(0, 1, 3)
            self.u3 = np.random.uniform(0, 1, 3)
            self.u1 = self.u1/np.sum(self.u1)
            self.u2 = self.u2/np.sum(self.u2)
            self.u3 = self.u3/np.sum(self.u3)

        vals = [0, 1, 2]
        i1 = np.random.choice(vals, p=self.u1)
        i2 = np.random.choice(vals, p=self.u1)
        i3 = np.random.choice(vals, p=self.u1)

        a1 = [0, 0.2, -0.2][i1]
        a2 = [0, 0.2, -0.2][i2]
        a3 = [0, 0.2, -0.2][i3]

        reward = 0
        ans = self.env.step([a1, a2, a3, 0])
        reward = reward + ans[3]["is_success"]
        self.obs = ans[0]["observation"]

        terminal = self.termChecker()
        self.numSteps += 1

        if terminal:
            return self.obs, reward, terminal, {}

        return self.obs, reward, terminal, {}

    def preProcess(self):
        raise NotImplementedError
