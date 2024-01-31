from ._base_optimizer import _BaseOptimizer
import numpy as np
class SGD(_BaseOptimizer):
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        super().__init__(learning_rate, reg)

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :param gradient: The Gradient computed in forward step
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)
        #############################################################################
        # TODO:                                                                     #
        #    1) Update model weights based on the learning rate and gradients       #
        #############################################################################
        
        # There's actually no stochastic term, so it's vanilla gradient descent

        # NOTE: model is already regularized (line 14)

        theta = model.weights
        grad_theta = model.gradients
        weights_keys = model.weights.keys()

        # Do that for every single weight
        for key in weights_keys:
            # print(key)
            theta[key] -= self.learning_rate * grad_theta[key]
            # model.weights[key] -= self.learning_rate * model.gradients[key]
        
        # # theta['W1'] -= self.learning_rate * grad_theta['W1']

        # # if 'W2' in weights_keys:
        # #     theta['W2'] -= self.learning_rate * grad_theta['W2']
        # #     theta['b1'] -= self.learning_rate * grad_theta['b1']
        # #     theta['b2'] -= self.learning_rate * grad_theta['b2']

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
