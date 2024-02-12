import torch
from torch.nn import Sequential as Seq, Dropout, LeakyReLU, Linear
from torch_geometric.nn import global_max_pool 
import torch.nn as nn
import torch.optim as optim

import deltaconv

#from deltaconv.models import DeltaNetBase
#from deltaconv.nn import MLP
import torch.nn.functional as F


class DeltaConvBasis(torch.nn.Module):
    def __init__(self, in_channels, k=30, conv_channels=[64, 128, 256], mlp_depth=2, embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1):
        
        super().__init__()
        self.k=k
        self.deltanet_base = deltaconv.models.DeltaNetBase(in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width)
        # Global embedding
        self.lin_embedding = deltaconv.nn.MLP([sum(conv_channels), embedding_size])
        #basis
        self.segmentation_head = Seq(
            deltaconv.nn.MLP([embedding_size+sum(conv_channels), 256]), Dropout(0.3),
            Linear(256, 128), Dropout(0.3), LeakyReLU(negative_slope=0.2), Linear(128, self.k))


    def forward(self, data):

        conv_out = self.deltanet_base(data)
        x = torch.cat(conv_out, dim=1)
        x = self.lin_embedding(x)

        batch = data.batch
        x_max = global_max_pool(x, batch)[batch]

        x = torch.cat([x_max] + conv_out, dim=1)

        return self.segmentation_head(x)


class DeltaConvDesc(torch.nn.Module):
    def __init__(self, in_channels, k=40, conv_channels=[64, 128, 256], mlp_depth=2, embedding_size=1024, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1):
        super().__init__()
        self.k=k
        self.deltanet_base = deltaconv.models.DeltaNetBase(in_channels, conv_channels, mlp_depth, num_neighbors, grad_regularizer, grad_kernel_width)
        # Global embedding
        self.lin_embedding = deltaconv.nn.MLP([sum(conv_channels), embedding_size])
        #basis
        self.segmentation_head = Seq(
            deltaconv.nn.MLP([embedding_size+sum(conv_channels), 256]), Dropout(0.3),  deltaconv.nn.MLP([256, 256]), Dropout(0.3),
            Linear(256, 128), LeakyReLU(negative_slope=0.2), Linear(128, self.k))


    def forward(self, data):

        conv_out = self.deltanet_base(data)

        x = torch.cat(conv_out, dim=1)
        x = self.lin_embedding(x)

        batch = data.batch
        x_max = global_max_pool(x, batch)[batch]

        x = torch.cat([x_max] + conv_out, dim=1)

        return self.segmentation_head(x)
