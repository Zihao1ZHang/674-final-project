import torch
from torch.nn import Linear

class SA_net(torch.nn.Module):
    def __init__(self, num_input, num_output):
        super(SA_net, self).__init__()
        self.fc = Linear(num_input, num_output)

    def forward(self, x):
        return self.fc(x)
