class _BaseOptimizer:
    def __init__(self, learning_rate=1e-4, reg=1e-3):
        self.learning_rate = learning_rate
        self.reg = reg

    def update(self, model):
        pass

    def apply_regularization(self, model):
        '''
        Apply L2 penalty to the model. Update the gradient dictionary in the model
        :param model: The model with gradients
        :return: None, but the gradient dictionary of the model should be updated
        '''

        #############################################################################
        # TODO:                                                                     #
        #    1) Apply L2 penalty to model weights based on the regularization       #
        #       coefficient                                                         #
        #############################################################################
        
        grad_theta = model.gradients
        gradients_keys = grad_theta.keys()
        # print(weights_keys)
        
        grad_theta['W1'] = grad_theta['W1'] * self.reg

        # DON'T regularize biases weights
        # ** Key assumption: only testing in softmax_relu and two_layer_nn AND/OR naming conventions are thye same**
        # If I get Gradescope errors, come back here and make it generalizable: test the sizes of matrices and skip the 1D ones

        if 'W2' in gradients_keys:
            grad_theta['W2'] = grad_theta['W2'] * self.reg


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################