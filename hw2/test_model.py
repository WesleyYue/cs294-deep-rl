import time
import unittest

import numpy as np
import tensorflow as tf

from model import PolicyGradient
from pprint import pprint


class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ob_dim = 5
        cls.ac_dim = 5
        cls.model = PolicyGradient(
            ob_dim=cls.ob_dim,
            ac_dim=cls.ac_dim,
            discrete=False,
            n_layers=2,
            size=32,
            learning_rate=5e-2,
            nn_baseline=True)

    def test_inspect_tensor_vars(self):
        try:
            print()
            self.model
        except AttributeError:
            self.setUpClass()

        for v in tf.trainable_variables():
            pprint(v.name)
            pprint(self.model.sess.run(v))
            pprint(v.shape)
            print("----------")

    def test_benchmark_training_time(self):
        try:
            print()
            self.model
        except AttributeError:
            self.setUpClass()

        BATCH_SIZE = 100
        observations = np.random.random((BATCH_SIZE, self.ob_dim))
        actions = np.random.random((BATCH_SIZE, self.ac_dim))
        advantages = np.random.random((BATCH_SIZE))

        start = time.time()

        self.model.train_agent(observations, actions, advantages)

        end = time.time()
        print(
            "\nBatch size of {batch_size} took {training_time:.2f}s to train".
            format(batch_size=BATCH_SIZE, training_time=(end - start)))
