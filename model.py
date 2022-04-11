import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self,input_size,hidden_size,num_classes):
        super(NeuralNet,self).__init__()
        #1st linear layer
        self.l1 = nn.Linear(input_size, hidden_size)
        #2nd linear layer
        self.l2 = nn.Linear(hidden_size,hidden_size)
        #3rd linear layer
        self.l3 = nn.Linear(hidden_size,num_classes)
        #activation function (relu)
        self.relu=nn.ReLU()

    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out = self.l2(x)
        out = self.relu(out)
        out = self.l3(x)
        #no activation function for the last layer
        return out



