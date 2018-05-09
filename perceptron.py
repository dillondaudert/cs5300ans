# Basic Perceptron class
import numpy as np

class Perceptron(object):

    def __init__(self):
        """
        Create and initialize a new Perceptron object.
        """
        pass

    def predict(self, x):
        """Predict the class of sample x. (Forward pass)"""
        pass

    def _delta(self, y_hat, y):
        """
        Given predictions y_hat and targets y, calculate the weight
        update delta.
        """
        pass

    def _update_weights(delta):
        """
        Update the weights by delta.
        """
        pass

    def _train_step(self, x, y):
        """
        Perform one training step:
            - predict
            - calculate delta
            - update weights
        Returns the predictions y_hat.
        """
        y_hat = self.predict(x)
        delta = self._delta(y_hat, y)
        self._update_weights(delta)
        return y_hat

    def train(self,
              train_x,
              train_y,
              num_steps):
        """
        Train the perceptron, performing num_steps weight updates.
        """
        pass

