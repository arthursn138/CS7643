import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        
        n, ch, h, w = x.shape
        H_out = int((h - self.kernel_size) / self.stride) + 1
        W_out = int((w - self.kernel_size) / self.stride) + 1

        # Output from this layer is a pooled image of size batch, chanels, H_out, W_out
        pooled_img = np.zeros((n, ch, H_out, W_out))

        # (keeping consistent with convolutional layer, will do horizontal pass in the outwards loop)
        # # aux_idx = np.zeros_like(x)  # found necessary to avoid another inner or outer loop
        for i in range(W_out):      # Passing horizontally, from right to left
            for j in range(H_out):  # Passing vertically, from top to bottom
                # Same strategy as in the convolutional layer: first get the top right and bottom left corners:
                k_w_0 = i * self.stride
                k_h_0 = j * self.stride
                k_w_end = k_w_0 + self.kernel_size
                k_h_end = k_h_0 + self.kernel_size

                # Get chunk of the image that's going to undergo the pooling operation:
                x_to_pool = x[:, :, k_h_0:k_h_end, k_w_0:k_w_end]

                # Pooling operation:
                x_to_pool = x_to_pool.reshape(n, ch, -1)    # reshape to get only the pixels fo a single channel in a single image
                max_idx = np.argmax(x_to_pool, axis=2)      # gets the position of the largest valued pixel in each image and channel
                # # print('# of images:', n, ' # of channels:', ch, 'h:', h, 'w:', w)
                # # print('the max idx is: ', max_idx)
                # # print('max idx(0): ', max_idx[0])

                # max_idx is n x c
                
                # Couldn't find a cleaner way to do it, so I'll loop over all images and channels
                for nn in range(n):
                    for k in range(ch):
                        pixel_idx = max_idx[nn, k]
                        pixel_value = x_to_pool[nn, k, pixel_idx]
                        pooled_img[nn, k, j, i] = pixel_value

                # # for k in range(ch):
                # #     pixel_idx = max_idx[:, k]
                # #     aux_idx[:, k, j, i] = pixel_idx
                # #     # print('pixel index', pixel_idx)
                # #     # max_pixel = x_to_pool[:, k, pixel_idx]
                # #     # print('val of pixel_idx: ', max_pixel)

                # # # x_to_pool = x_to_pool.reshape(n, ch, h, w)
                # # print(aux_idx)
                # # pooled_img[:, :, j, i] = x_to_pool[:, :, aux_idx]

                # # max_pixel = x_to_pool[:, :, max_idx]

        out = pooled_img

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        
        # print('dout shape', dout.shape)
        # print('H', H_out, 'W', W_out)
        # # print(np.unravel_index(dout, (H_out, W_out)))
        # print('x shape', x.shape)
        n, ch, h, w = x.shape

        # (keeping consistent with convolutional layer, will do horizontal pass in the outwards loop)
        dx = np.zeros_like(x)
        for i in range(W_out):
            for j in range(H_out):
                # Get corners
                k_w_0 = i * self.stride
                k_h_0 = j * self.stride
                k_w_end = k_w_0 + self.kernel_size
                k_h_end = k_h_0 + self.kernel_size

                # # Pooling operation:
                # x_from_pool = x[:, :, k_h_0:k_h_end, k_w_0:k_w_end] # Window with all pixels that touched the one the grad is taken w.r.t.
                # max_idx = np.argmax(x_from_pool)
                # idx_h, idx_w = np.unravel_index(max_idx, (self.kernel_size, self.kernel_size))

                # Accumulates grads (Lec 10)
                # # dx[:, :, j + idx_h, i + idx_w] = dout[:, :, j:(j + 1), i:(i + 1)]
                dx[:, :, k_h_0:k_h_end, k_w_0:k_w_end] += dout[:, :, j:(j + 1), i:(i + 1)] #* x[j, i]

        self.dx = dx

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
