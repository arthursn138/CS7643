import numpy as np

class Conv2D:
    '''
    An implementation of the convolutional layer. We convolve the input with out_channels different filters
    and each filter spans all channels in the input.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        '''
        :param in_channels: the number of channels of the input data
        :param out_channels: the number of channels of the output (aka the number of filters applied in the layer)
        :param kernel_size: the specified size of the kernel (both height and width)
        :param stride: the stride of convolution
        :param padding: the size of padding. Pad zeros to the input with padding size.
        '''
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.cache = None

        self._init_weights()

    def _init_weights(self):
        np.random.seed(1024)
        self.weight = 1e-3 * np.random.randn(self.out_channels, self.in_channels,  self.kernel_size, self.kernel_size)
        self.bias = np.zeros(self.out_channels)

        self.dx = None
        self.dw = None
        self.db = None

    def forward(self, x):
        '''
        The forward pass of convolution
        :param x: input data of shape (N, C, H, W)
        :return: output data of shape (N, self.out_channels, H', W') where H' and W' are determined by the convolution
                 parameters. Save necessary variables in self.cache for backward pass
        '''
        out = None
        #############################################################################
        # TODO: Implement the convolution forward pass.                             #
        # Hint: 1) You may use np.pad for padding.                                  #
        #       2) You may implement the convolution with loops                     #
        #############################################################################
        
        n, ch, h, w = x.shape
        # print(x.shape)
        # print(self.padding)
        
        ## Padding the image 
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        # print(x_pad.shape)

        ## Feature map output shape (ch = out_channels):
        # print(self.kernel_size, type(self.kernel_size))
        # print(self.stride)
        h_out = int((h + 2 * self.padding - self.kernel_size) / self.stride) + 1    # 2x padding bc pads two edges
        w_out = int((w + 2 * self.padding - self.kernel_size) / self.stride) + 1    # 2x padding bc pads two edges
        
        # Doing everything w.r.t. the output size (feature map out size (n, self.out_channels, h_out, w_out)):
        feature_map_weights = np.zeros((n, self.out_channels, h_out, w_out))

        # Convolution loop
        for i in range(w_out):      # Passing horizontally, from left to right
            for j in range(h_out):  # Passing vertically, from top to bottom --- In hindsight it'd be less confusing doing vertical first, then horizontal (i would come before j)
                ## Filter pass (convolution):
                # Start of the filter (top right corner)
                k_w_0 = i * self.stride   # Horizontal localization of the top corner of a window that's going to be convolved
                k_h_0 = j * self.stride   # Vertical localization of the top corner of a window that's going to be convolved
                # End of the filter (bottom right corner)
                k_w_end = k_w_0 + self.kernel_size   # Horizontal localization of the bottom corner of a window that's going to be convolved
                k_h_end = k_h_0 + self.kernel_size   # Horizontal localization of the bottom corner of a window that's going to be convolved
                # Opperation (Reminder: dims: 2=ch; 3=height; 4=widht)
                feature_map_weights[:, :, j, i] = np.sum(x_padded[:, np.newaxis, :, k_h_0:k_h_end, k_w_0:k_w_end] * self.weight[np.newaxis, :, :, :], axis=(2,3,4))

        # Update weights
        out = feature_map_weights + self.bias[np.newaxis, :, np.newaxis, np.newaxis]    # Bias in ch only

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = x
        return out

    def backward(self, dout):
        '''
        The backward pass of convolution
        :param dout: upstream gradients
        :return: nothing but dx, dw, and db of self should be updated
        '''
        x = self.cache
        #############################################################################
        # TODO: Implement the convolution backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the convolution with loops                     #
        #       2) don't forget padding when computing dx                           #
        #############################################################################
        
        # print(dout.shape)
        # print(x.shape)
        
        n, ch, hx, wx = x.shape
        _, _, hdout, wdout = dout.shape
        
        # For backpass, we know that the final sizes are the weights sizes
        hfinal, wfinal = self.weight.shape[2], self.weight.shape[3]
        
        x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant') # Will be used for calculating dx, so needs padding, as per instructions

        weights = np.zeros_like(self.weight)
        dx = np.zeros_like(x_padded)
        
        for i in range(wdout):      # Passing horizontally, from right to left (like in the fwd pass)
            for j in range(hdout):  # Passing vertically, from top to bottom (like in the fwd pass)
                
                ## Same strategy as in the fwd pass: get corners and perform convolution (but this time it's gradients)
                # Getting corners:
                k_w_0 = i * self.stride
                k_h_0 = j * self.stride
                k_w_end = k_w_0 + wfinal
                k_h_end = k_h_0 + hfinal

                # Accumulating grads in dx and w (cross-correlation between upstream grad and input - Lec10):
                dx[:, :, k_h_0:k_h_end, k_w_0:k_w_end] += np.sum(self.weight[np.newaxis, :, :, :] * dout[:, :, j:j+1, i:i+1, np.newaxis], axis=1)
                weights += np.sum(x_padded[:, np.newaxis, :, k_h_0:k_h_end, k_w_0:k_w_end] * dout[:, :, j:j+1, i:i+1, np.newaxis], axis=0)
        
        # print('hx', hx, 'wx', wx)
        self.dx = dx[:, :, self.padding:self.padding + hx, self.padding:self.padding + wx]

        self.dw = weights

        d_biases = dout.sum(axis=(0, 2, 3))
        self.db = d_biases

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################