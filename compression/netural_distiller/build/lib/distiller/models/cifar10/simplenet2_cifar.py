"""
student net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__=["simplenet2_cifar"]

class Simplenet(nn.Module):
    def __init__(self):
        super(Simplenet, self).__init__()
        self.conv1=nn.Conv2d(3,16,7,1,4)
        self.pool1=nn.MaxPool2d(2,2)
        self.relu_conv1=nn.ReLU()
        self.conv2=nn.Conv2d(16,64,3,2)
        self.relu_conv2=nn.ReLU()
        self.fc1=nn.Linear(64*8*8,10)
        self.relu_fc1=nn.ReLU()


    def forward(self, x):
        x=self.pool1(self.relu_conv1(self.conv1(x)))
        x=self.relu_conv2(self.conv2(x))
        x=x.view(-1,64*8*8)
        x=self.relu_fc1(self.fc1(x))
        return x

def simplenet2_cifar():
    model=Simplenet()
    return model


if __name__=="__main__":
    x=torch.rand(2,3,32,32)
    net=Simplenet()
    x=net(x)
    print(x)

