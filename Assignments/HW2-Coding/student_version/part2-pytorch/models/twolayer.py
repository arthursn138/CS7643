import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        
        self.input_size = input_dim
        self.output_size = num_classes
        self.fc1 = nn.Linear(self.input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, self.output_size)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        
        hidden1 = self.fc1(x)           # 1st activation
        sigmoid = self.sigmoid(hidden1) # Sigmoid (2nd) activation
        hidden2 = self.fc2(sigmoid)     # 2nd layer (3rd) activation = scores

        out = hidden2

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out