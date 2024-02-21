from ._base_optimizer import _BaseOptimizer
class SGD(_BaseOptimizer):
    def __init__(self, model, learning_rate=1e-4, reg=1e-3, momentum=0.9):
        super().__init__(model, learning_rate, reg)
        self.momentum = momentum

        # initialize the velocity terms for each weight

    def update(self, model):
        '''
        Update model weights based on gradients
        :param model: The model to be updated
        :return: None, but the model weights should be updated
        '''
        self.apply_regularization(model)

        for idx, m in enumerate(model.modules):
            if hasattr(m, 'weight'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for weights                                        #
                #############################################################################
                
                ## The following is modified from my HW1:
                # theta = model.weight
                # theta = m.dw
                # grad_theta = model.dw
                # weights_keys = model.weight.keys()

                # # Do that for every single weight --- Already in a for loop!
                # for key in weights_keys:
                #     # print(key)
                #     theta[key] -= self.learning_rate * grad_theta[key]
                

                # Using self.grad_tracker from _BaseOptimizer:
                velocity = self.momentum * self.grad_tracker[idx]['dw']                 # velocity term (beta * v_{t-1})
                # self.grad_tracker[idx]['dw'] = velocity - self.learning_rate * theta    # GD with velocity (v_t)
                # theta = m.weight + self.grad_tracker[idx]['dw']                         # Weight update
                # m.dw = theta
                
                self.grad_tracker[idx]['dw'] = velocity - self.learning_rate * m.dw    # GD with velocity (v_t)
                m.weight = m.weight + self.grad_tracker[idx]['dw']                         # Weight update

                

                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################
            if hasattr(m, 'bias'):
                #############################################################################
                # TODO:                                                                     #
                #    1) Momentum updates for bias                                           #
                #############################################################################
                
                # theta_b = m.db
                # Using self.grad_tracker from _BaseOptimizer:
                velocity = self.momentum * self.grad_tracker[idx]['db']                 # velocity term (beta * v_{t-1})
                # self.grad_tracker[idx]['db'] = velocity - self.learning_rate * theta_b  # GD with velocity (v_t)
                # theta_b = m.bias + self.grad_tracker[idx]['db']                         # Weight update

                # m.db = theta_b

                self.grad_tracker[idx]['db'] = velocity - self.learning_rate * m.db  # GD with velocity (v_t)
                m.bias = m.bias + self.grad_tracker[idx]['db']                         # Weight update

                #############################################################################
                #                              END OF YOUR CODE                             #
                #############################################################################