import torch
import torch.nn as nn


class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        
        # Beggining like the vanilla CNN

        self.in_ch = 3 # Relying on CIFAR-10 (32x32)
        self.out_ch = 32
        self.kernel_size_conv = 7
        self.stride_conv = 1
        self.padding = 0
        
        self.kernel_size_pool = 2
        self.stride_pool = 2

        self.out_size = 10
        self.hidden_size_1 = 128
        self.hidden_size_2 = 64
        self.hidden_size_3 = 32

        # self.conv = nn.Conv2d(self.in_ch, self.out_ch, self.kernel_size_conv, self.stride_conv, self.padding)
        self.conv = nn.Conv2d(self.in_ch, 16, 5, self.stride_conv, 0)
        self.conv2 = nn.Conv2d(16, 16, 3, 1, 0)
        self.conv3 = nn.Conv2d(16, 3, 3, 1, 0)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(self.kernel_size_pool, self.stride_pool)

        # http://layer-calc.com/ with CIFAR-10 (32x32), above conv2d and maxpool2d: (32, 13, 13)
        # self.maxpool_out_size = 5408
        # self.maxpool_out_size = 576
        self.maxpool_out_size = 16 * 14 * 14
        
        self.fc1 = nn.Linear(self.maxpool_out_size, self.hidden_size_1)
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.fc3 = nn.Linear(self.hidden_size_2, self.hidden_size_2)
        self.fc4 = nn.Linear(self.hidden_size_2, self.out_size)        
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        out1 = self.relu(self.conv(x))          # Convolution w/ 16ch out + relu activation
        # out2 = self.relu(self.conv2(out1))      # Convolution w/ 16ch out + relu activation
        # pool_out = self.maxpool(out1)       # Maxpooling
        # out2 = self.relu(self.conv3(out1)) 
        # pool2 = self.maxpool(out2)          # Maxpooling

        pool_out = self.maxpool(out1)       # Maxpooling

        # out3 = self.relu(self.conv2(pool_out))  # Conv 3ch out + relu activation ()
        # out4 = self.relu(self.conv2(out3))      # Conv 3ch out + relu activation ()
        
        hidden1 = self.relu(self.fc1(torch.flatten(pool_out, 1)))
        hidden2 = self.relu(self.fc2(hidden1))
        hidden3 = self.relu(self.fc3(hidden2))
        outs = self.relu(self.fc4(hidden2))
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs