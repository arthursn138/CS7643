import torch
import torch.nn as nn


class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self, input_size=10, hidden_size=32, fc2_size=32):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        
        self.input_size = input_size
        self.output_size = input_size
        self.fc1 = torch.nn.Linear(self.input_size, hidden_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(hidden_size, self.output_size)
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################

        hidden1 = self.fc1(x)
        sigmoid = self.sigmoid(hidden1)
        hidden2 = self.fc2(sigmoid)

        outs = hidden2
        
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs