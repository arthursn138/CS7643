from .softmax_ce import SoftmaxCrossEntropy
from .relu import ReLU
from .max_pool import MaxPooling
from .convolution import Conv2D
from .linear import Linear


class ConvNet:
    '''
    Max Pooling of input
    '''
    def __init__(self, modules, criterion):
        self.modules = []
        for m in modules:
            if m['type'] == 'Conv2D':
                self.modules.append(
                    Conv2D(m['in_channels'],
                           m['out_channels'],
                           m['kernel_size'],
                           m['stride'],
                           m['padding'])
                )
            elif m['type'] == 'ReLU':
                self.modules.append(
                    ReLU()
                )
            elif m['type'] == 'MaxPooling':
                self.modules.append(
                    MaxPooling(m['kernel_size'],
                               m['stride'])
                )
            elif m['type'] == 'Linear':
                self.modules.append(
                    Linear(m['in_dim'],
                           m['out_dim'])
                )
        if criterion['type'] == 'SoftmaxCrossEntropy':
            self.criterion = SoftmaxCrossEntropy()
        else:
            raise ValueError("Wrong Criterion Passed")

    def forward(self, x, y):
        '''
        The forward pass of the model
        :param x: input data: (N, C, H, W)
        :param y: input label: (N, )
        :return:
          probs: the probabilities of all classes: (N, num_classes)
          loss: the cross entropy loss
        '''
        probs = None
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement forward pass of the model                                 #
        #############################################################################
        
        # # print(self.modules)
        input = x   # First input has to be the actual data x
        for i in self.modules:
            # print('This is an element in "slef.modules":', i)
            out_hidden = i.forward(input)   # Passes last output as input to the current layer
            input = out_hidden              # Current output becomes input to the next

        probs, loss = self.criterion.forward(input, y)  # Simple fwd pass in the function defined in softmax_ce.py

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return probs, loss

    def backward(self):
        '''
        The backward pass of the model
        :return: nothing but dx, dw, and db of all modules are updated
        '''
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement backward pass of the model                                #
        #############################################################################
        
        # Go each step back at a time (contrary of above)
        self.criterion.backward()       # Takes no argument, has dx
        last_out = self.criterion.dx    # Gets what is going to be fed to the previous layers ("upstream" gradients)

        for i in reversed(self.modules): #.reverse(): (doesn't work)
            # # print(i)
            i.backward(last_out)    # All have dx, some have db and dw. Doing a back pass satisfies updating those grads
            last_out = i.dx         # Gradients were updated above, dx is the appropriate input to the previous layer

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

# # # # # # ## TO TEST:
# # # # # # import numpy as np

# # # # # # x = np.array((1,2,3),(1,3,4),(1,2,3))
# # # # # # y = np.array(0, 1, 2)

# # # # # # net = ConvNet()
# # # # # # net.forward(x, y)