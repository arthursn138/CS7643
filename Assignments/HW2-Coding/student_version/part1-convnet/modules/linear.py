import numpy as np

class Linear:
    '''
    A linear layer with weight W and bias b. Output is computed by y = Wx + b
    '''
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.in_dim, self.out_dim)
        np.random.seed(1024)
        self.bias = np.zeros(self.out_dim)

        self.dx = None
        self.dw = None
        self.db = None


    def forward(self, x):
        '''
        Forward pass of linear layer
        :param x: input data, (N, d1, d2, ..., dn) where the product of d1, d2, ..., dn is equal to self.in_dim
        :return: The output computed by Wx+b. Save necessary variables in cache for backward
        '''
        out = None
        #############################################################################
        # TODO: Implement the forward pass.                                         #
        #    HINT: You may want to flatten the input first                          #
        #############################################################################
        
        # print(x.shape) # total data, ch, height, widht
        # print(self.in_dim)
        n, *_ = x.shape
        x_flat = x.reshape(n, -1)
        out = np.matmul(x_flat, self.weight) + self.bias

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        Computes the backward pass of linear layer
        :param dout: Upstream gradients, (N, self.out_dim)
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the linear backward pass.                                 #
        #############################################################################
        
        # print(x.shape)
        # n, h, w = x.shape # DOESN'T WORK GOOD WITH MONO AND NON MONO CHROMATIC
        # # print(np.shape(x.shape))
        # n = x.shape[0]
        # h = x.shape[-2]
        # w = x.shape[-1]
        shape_tuple = x.shape   # much cleaner
        self.dx = np.matmul(dout, self.weight.T).reshape(shape_tuple)  # grad loss wrt inputs/previous layer
        x_flat = x.reshape(x.shape[0], -1)
        # print(x.shape, x_flat.shape, n*m)
        self.dw = np.matmul(x_flat.T, dout) # grad loss wrt weights (W)
        self.db = np.sum(dout, axis=0)      # grad loss wrt weights (b)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
