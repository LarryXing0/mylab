"""
mmd_loss

"""

import torch
import torch.nn as nn
import random
from torch.autograd import Variable

class MMD_Loss(nn.Module):
    def __init__(self,kernel_mul=2.0,kernel_num=5):
        super(MMD_Loss,self).__init__()
        self.kernel_num=kernel_num
        self.kernel_mul=kernel_mul
        self.fix_sigma=None

    def guassian_kernel(self,source,target,kernel_mul=2,kernel_num=5,fix_sigma=None):
        n_samples=int(source.size()[0])+int(target.size()[0])
        total=torch.cat([source,target],dim=0)
        total0=total.unsqueeze(0).expand(int(total.size(0)),int(total.size(0)),int(total.size(1)))
        total1=total.unsqueeze(1).expand(int(total.size(0)),int(total.size(0)),int(total.size(1)))
        L2_distance=((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth=fix_sigma
        else:
            bandwidth=torch.sum(L2_distance.data)/(n_samples**2-n_samples)
        bandwidth/=kernel_mul**(kernel_num//2)
        bandwidth_list=[bandwidth*(kernel_mul**i) for i in range(kernel_num)]
        kernel_val=[torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source,target):
        batch_size=int(source.size()[0])
        kernels=self.guassian_kernel(source,target,kernel_mul=self.kernel_mul,kernel_num=self.kernel_num,fix_sigma=self.fix_sigma)
        XX=kernels[:batch_size,:batch_size]
        YY=kernels[batch_size:,batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss=torch.mean(XY+YY-XY-YX)
        return loss

if __name__=="__main__":
    SAMPLE_SIZE=500
    buckets=50
    mu=0.6
    sigma=0.15
    aplha=1
    beta=10
    diff_1=[]
    for i in range(10):
        diff_1.append([random.lognormvariate(mu,sigma) for _ in range(1,SAMPLE_SIZE)])
    diff_2=[]
    for j in range(10):
        diff_2.append([random.betavariate(aplha,beta) for _ in range(1,SAMPLE_SIZE)])
    x=torch.Tensor(diff_1)
    y=torch.Tensor(diff_2)
    x,y=Variable(x),Variable(y)
    print(MMD_Loss().forward(x,y))