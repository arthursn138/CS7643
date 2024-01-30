# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()


    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)

        # initialize gradients to zeros
        self.gradients['W1'] = np.zeros((self.input_size, self.hidden_size))
        self.gradients['b1'] = np.zeros(self.hidden_size)
        self.gradients['W2'] = np.zeros((self.hidden_size, self.num_classes))
        self.gradients['b2'] = np.zeros(self.num_classes)

    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        loss = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################

        z1 = np.matmul(X, self.weights['W1']) + self.weights['b1']      # Shape: N, hidden_size
        sig = _baseNetwork.sigmoid(_baseNetwork, z1)                    # Shape: N, hidden_size
        z2 = np.matmul(sig, self.weights['W2']) + self.weights['b2']    # Shape: N, num_classes
        scores = _baseNetwork.softmax(_baseNetwork, z2)                 # Shape: N, num_classes

        loss = _baseNetwork.cross_entropy_loss(_baseNetwork, scores, y)
        accuracy = _baseNetwork.compute_accuracy(_baseNetwork, scores, y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################

        # From HW1 Tutorial and link provided (https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
        grad_CEL_softmax = np.copy(scores)
        for i in range(len(y)):
            grad_CEL_softmax[i, y[i]] -= 1
        grad_CEL_softmax = grad_CEL_softmax / len(y)    # Shape: N, num_classes

        grad_L2_w2 = np.matmul(sig.T, grad_CEL_softmax) # Shape: (hidden, N x N, num_classes) = hidden_size, num_classes
        grad_CEL_b2 = np.matmul(np.ones(np.shape(X)[0]),
                                grad_CEL_softmax)       # Shape: (num_classes,)
        # print('Hidden size:', self.hidden_size)
        # print(grad_CEL_b2.shape)
        # print('input size:', self.input_size)
        # print('num classes', self.num_classes)

        self.gradients['W2'] = grad_L2_w2
        self.gradients['b2'] = grad_CEL_b2

        grad_L2_sig = self.weights['W2']        # Shape: hidden_size, num_classes
        grad_sig_L1 = _baseNetwork.sigmoid_dev(_baseNetwork, z1)    # Shape: N, hidden_size
        grad_L1_w1 = X.T                        # Shape: input_size, N
        grad_L1_b1 = np.ones(np.shape(X)[0])    # Shape: (N,)

        # # # grad_sig_w1 = np.matmul(grad_L1_w1, grad_sig_L1)            # Shape: input_size, hidden_size
        grad_CEL_sig = np.matmul(grad_CEL_softmax, grad_L2_sig.T)   # Shape: N, hidden_size
        # # # grad_sig_b1 = np.matmul(grad_L1_b1, grad_sig_L1)            # Shape: (hidden_size,)

        # # Check sizes:
        # print(f'grad_L1_w1:{grad_L1_w1.shape}; grad_sig_L1:{grad_sig_L1.shape} -> grad_sig_w1:{grad_sig_w1.shape}')
        # print(f'grad_CEL_softmax:{grad_CEL_softmax.shape}; grad_L2_sig.T:{grad_L2_sig.T.shape} -> grad_CEL_sig:{grad_CEL_sig.shape}')
        # # print(f'grad_L1_b1:{grad_L1_b1.shape}; grad_sig_L1:{grad_sig_L1.shape} -> grad_sig_b1:{grad_sig_b1.shape}\ngrad_CEL_sig:{grad_CEL_sig.shape}')
        # print(f'grad_CEL_sig:{grad_CEL_sig.shape}; grad_sig_b1:{grad_sig_b1.shape}')

        self.gradients['W1'] = np.matmul(grad_L1_w1, grad_CEL_sig * grad_sig_L1)    # Shape: input_size, hidden_size
        self.gradients['b1'] = np.matmul(grad_L1_b1, grad_CEL_sig * grad_sig_L1)    # Shape: (hidden_size,)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


        return loss, accuracy


