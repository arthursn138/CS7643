import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        
        self.in_ch = 3 # Relying on CIFAR-10 (32x32)
        self.out_ch = 32
        self.kernel_size_conv = 7
        self.stride_conv = 1
        self.padding = 0
        
        self.kernel_size_pool = 2
        self.stride_pool = 2

        self.out_size = 10

        self.conv = nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size_conv, self.stride_conv, self.padding)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(self.kernel_size_pool, self.stride_pool)

        # http://layer-calc.com/ with CIFAR-10 (32x32), above conv2d and maxpool2d: (32, 13, 13)
        self.maxpool_out_size = 5408
        self.fc = nn.Linear(self.maxpool_out_size, self.out_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        
        conv = self.conv(x)
        relu = self.relu(conv)
        maxpool = self.maxpool(relu)
        hidden = torch.flatten(maxpool, 1)
        outs = self.fc(hidden)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs