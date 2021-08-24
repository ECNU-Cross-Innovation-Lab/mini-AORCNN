import torch
from torch import nn
import numpy as np
from torchstat import stat

class ARFRes(nn.Module):
    def __init__(self,in_c,out_c):
        super(ARFRes,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_c,out_channels = in_c,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(in_channels = in_c,out_channels = in_c,kernel_size = 3,padding = 1),
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
        )
        self.botneck = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        
    def forward(self,x):
        x_prim = x
        x = self.conv(x)
        x = self.botneck(x + x_prim)
        x = self.pool(x)
        return x

class ARFSA(nn.Module):
    def __init__(self,in_c,out_c,fm_sz,pos_bias = False):
        super(ARFSA,self).__init__()
        self.w_q = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.w_k = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.w_v = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.pos_code = self.__getPosCode(fm_sz,out_c)
        self.softmax = nn.Softmax(dim = 2)
        self.pos_bias = pos_bias

    def __getPosCode(self,fm_sz,out_c):
        x = []
        for i in range(fm_sz):
            x.append([np.sin,np.cos][i % 2](1 / (10000 ** (i // 2 / fm_sz))))
        x = torch.from_numpy(np.array([x])).float()
        return torch.cat([(x + x.t()).unsqueeze(0) for i in range(out_c)])
    
    def forward(self,x):
        q,k,v = self.w_q(x),self.w_k(x),self.w_v(x)
        pos_code = torch.cat([self.pos_code.unsqueeze(0) for i in range(x.shape[0])]).to(x.device)
        if self.pos_bias:
            att_map = torch.matmul(q,k.permute(0,1,3,2)) + pos_code
        else:    
            att_map = torch.matmul(q,k.permute(0,1,3,2)) + torch.matmul(q,pos_code.permute(0,1,3,2))
        am_shape = att_map.shape
        att_map = self.softmax(att_map.view(am_shape[0],am_shape[1],am_shape[2] * am_shape[3])).view(am_shape)
        return att_map * v

class ARFMHSA(nn.Module):
    def __init__(self,in_c,out_c,head_n,fm_sz,pos_bias = False):
        super(ARFMHSA,self).__init__()
        self.sa_blocks = [ARFSA(in_c = in_c,out_c = out_c,fm_sz = fm_sz,pos_bias = pos_bias) for i in range(head_n)]
        self.sa_blocks = nn.ModuleList(self.sa_blocks)
        
    def forward(self,x):
        results = [sa(x) for sa in self.sa_blocks]
        return torch.cat(results,dim = 1)

class ARFBottleneckTransformer(nn.Module):
    def __init__(self,in_c,out_c,fm_sz,net_type = 'mhsa',head_n = 4):
        super(ARFBottleneckTransformer,self).__init__()
        self.botneck = nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = 1)
        self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
        if net_type == 'mhsa':
            self.sa = nn.Sequential(
                ARFMHSA(in_c = in_c,out_c = out_c // head_n,head_n = head_n,fm_sz = fm_sz),
                ARFMHSA(in_c = out_c,out_c = out_c // head_n,head_n = head_n,fm_sz = fm_sz)
            )
        elif net_type == 'sa':
            self.sa = nn.Sequential(
                ARFSA(in_c = in_c,out_c = out_c,fm_sz = fm_sz),
                ARFSA(in_c = out_c,out_c = out_c,fm_sz = fm_sz)
            )
    
    def forward(self,x):
        x0 = self.botneck(x)
        x = self.sa(x)
        x = x + x0
        x = self.pool(x)
        return x
        
class ARFNet(nn.Module):
    def __init__(self,net_type = 'mhsa'):
        super(ARFNet,self).__init__()
        #[N,2,112,112]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = 3,out_channels = 8,kernel_size = 7,stride = 2,padding = 3),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            #nn.MaxPool2d(kernel_size = 3,stride = 2,padding = 1)
        )
        #[N,4,56,56]
        self.res1 = ARFRes(in_c = 8,out_c = 18)
        #[N,8,28,28]
        self.res2 = ARFRes(in_c = 18,out_c = 28)
        #[N,16,14,14]
        if net_type == 'res':
            self.sa = ARFRes(in_c = 28,out_c = 8)
        else:
            self.sa = ARFBottleneckTransformer(in_c = 28,out_c = 8,fm_sz = 14,net_type = net_type)
        #[N,8,14,14]
        #[N,8,7,7]
        self.dense = nn.Sequential(
            nn.Linear(392,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,3)
        )
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.sa(x)
        x = x.view(-1,392)
        x = self.dense(x)
        return x

if __name__ == '__main__':
    stat(ARFNet(),(3,112,112))