# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork

class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28*28, num_classes=10):
        '''
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => ReLU activation => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        '''
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        '''
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        '''
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################

        # x_flat = X.flatten()    # shape: N*28*28
        # # print('KEYS', self.weights.keys())
        # # print('SHAPE OF W', self.weights['W1'].shape)
        # # print('X: ', X.shape)
        # # print('Y:', y)
        
        linear_layer = np.dot(X, self.weights['W1']) # shape: N,784 x 784,10 = N, num_classes
        # # print('CHECK SHAPE OF DOT PRODUCT:', linear_layer.shape)
        relu = _baseNetwork.ReLU(_baseNetwork, linear_layer)    # shape: N, num_classes
        # # print('SHAPE OF ReLU:', relu.shape) 
        softmax = _baseNetwork.softmax(_baseNetwork, relu)  # shape: N, num_classes
        # # print('SHAPE OF softmax:', softmax.shape)
        loss = _baseNetwork.cross_entropy_loss(_baseNetwork, softmax, y)    # shape: N, num_classes
        # # print('LOSS SHAPE:', loss.shape, 'VALUE:', loss)
        accuracy = _baseNetwork.compute_accuracy(_baseNetwork, softmax, y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        
        # From class: matmul order matters! Need to go from last-most to first layers

        # From Gradient notes (in L6's slides): dL/dW = < dL/dOut , dOut/dW > (typo in there)
        # In this case: dL/dW = dL/dSoftmax @ dSoftmax/dRelu @ dRelu/dLinear @ dLinear/dW
        # N,10 x N,10 x N,10 x N,784

        # # grad_linear_x = self.weights
        grad_linear_w = X
        # # print('GRAD_Linear', grad_linear_w.shape)
        
        # # grad_L = np.argmax(softmax, axis=1) - loss   # chosen label - y
        # # grad_softmax = relu # effectively it's the scores before being transformed into probabilities

        # From HW1 Tutorial and link provided (https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
        grad_L = (softmax - loss) / len(y)   # shape: N, num_classes ---> Unsure if should thake the "average" (i.e., divide by N)
        # # print('GRAD_L', grad_L.shape)
        
        grad_relu = _baseNetwork.ReLU_dev(_baseNetwork, linear_layer)   # shape: N, num_classes
        # # print('grad ReLU:', np.shape(grad_relu))

        # From HW1 Tutorial: dL/dW = dL/dRelu * dRelu/dLinear * dLinear/dW ---> N,10 * N,10 x N,784
        # Trick to not need to compute deriv. of softmax:
        grad_L_linear = grad_L * grad_relu
        # # print('Shape dLoss/dLinear:', grad_L_linear.shape)

        gradient = np.matmul(grad_linear_w.T, grad_L_linear)
        # # print('Gradient shape:', gradient.shape)

        self.gradients['W1'] = gradient

        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy





        


