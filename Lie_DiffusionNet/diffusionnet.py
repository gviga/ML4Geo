from __future__ import print_function
import torch
#pacchetto torch.nn per creare la rete di convoluzione
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import diffusion_net

############ BASIS NETWORK

#costruisco le architetture

class DiffusionNetBasis(nn.Module):
    def __init__(self, k = 30, n_block=12, feature_transform=False):
        super(DiffusionNetBasis, self).__init__()
        self.k = k
        self.diffusion=diffusion_net.layers.DiffusionNet(C_in=3,C_out=k,C_width=128,N_block=n_block, dropout=True)


    def forward(self,  x,mass,lap,evals,evecs,gradx,grady):
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        x = self.diffusion(x,mass,lap,evals,evecs,gradx,grady)
        x = x.view(batchsize, n_pts, self.k)
    
        return x

############ DESC NETWORK
class DiffusionNetDesc(nn.Module):
    def __init__(self, k = 30, n_block=12, feature_transform=False):
        super(DiffusionNetDesc, self).__init__()
        self.k = k
        self.diffusion=diffusion_net.layers.DiffusionNet(C_in=3,C_out=k,C_width=128,N_block=n_block, dropout=True)


    def forward(self,  x,mass,lap,evals,evecs,gradx,grady):
        batchsize = x.size()[0]
        n_pts = x.size()[1]
        x = self.diffusion(x,mass,lap,evals,evecs,gradx,grady)
        x = x.view(batchsize, n_pts, self.k)
    
        return x
