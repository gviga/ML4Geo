from __future__ import print_function
import argparse
import os
import random
import torch
import torch.optim as optim
from diffusionnet import DiffusionNetBasis
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from dataload_diff import Surr12kModelNetDataLoader as DataLoader
from tqdm import tqdm
from diffusion_net import geometry


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
manualSeed = 1  # fix seed
random.seed(manualSeed)
torch.manual_seed(manualSeed)

b_size = 8

# Out Dir
outf = './models/diffusionnet/'
try:
    os.makedirs(outf)
except OSError:
    pass


DATA_PATH = 'data/'


TRAIN_DATASET = DataLoader(root=DATA_PATH, npoint=1000,k_eig=40, split='train',
                                                    normal_channel=False, augm = True)
TEST_DATASET = DataLoader(root=DATA_PATH, npoint=1000,k_eig=40, split='test',
                                                    normal_channel=False, augm = True)

dataset = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=b_size, shuffle=True, num_workers=0)
dataset_test = torch.utils.data.DataLoader(TEST_DATASET, batch_size=b_size, shuffle=True, num_workers=0)

# BasisNetwork with 20 basis
basisNet = DiffusionNetBasis(k=30,n_block=12, feature_transform=False)

# Optimizer
optimizer = optim.Adam(basisNet.parameters(), lr=0.0001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)
basisNet.cuda()

best_eval_loss = np.inf;

train_losses = [];
eval_losses = [];
faust_losses = [];



# Training Loop
for epoch in range(400):
    scheduler.step()
    train_loss = 0
    # Training single Epoch
    for data in tqdm(dataset, 0):
        points = data[0]
        mass= data[1]
        evecs=data[4][:,:,:50]
        evals= data[3][:,:50]
        lap=data[2]
        gradx=data[5]
        grady=data[6]
        points = points.cuda().to(device)
        mass=mass.cuda().to(device)
        evecs=evecs.cuda().to(device)
        evals=evals.cuda().to(device)
        gradx=gradx.cuda().to(device)
        grady=grady.cuda().to(device)

        optimizer.zero_grad()
        basisNet = basisNet.train()

        # Obtaining predicted basis
        pred = basisNet(points,mass,lap,evals,evecs,gradx,grady)

        # Generating pairs
        basis_A = pred[1:,:,:]; basis_B = pred[:-1,:,:] 
        pc_A = points[1:,:,:]; pc_B = points[:-1,:,:]
        # Computing optimal transformation
        pseudo_inv_A = torch.pinverse(basis_A)
        C_opt = torch.matmul(pseudo_inv_A, basis_B)
        opt_A = torch.matmul(basis_A, C_opt)

        # SoftMap
        dist_matrix = torch.cdist(opt_A, basis_B)       
        s_max = torch.nn.Softmax(dim=1)
        s_max_matrix = s_max(-dist_matrix)

        # Basis Loss
        eucl_loss = torch.sum(torch.square(torch.matmul(s_max_matrix, pc_B) - pc_B))
        
        # Back Prop
        eucl_loss.backward()
        optimizer.step()
        train_loss += eucl_loss.item()
    
    # Validation
    with torch.no_grad():
        eval_loss = 0
        for data in tqdm(dataset_test, 0):
            points = data[0]
            mass= data[1]
            evecs=data[4][:,:,:50]
            evals= data[3][:,:50]
            lap=data[2]
            gradx=data[5]
            grady=data[6]
            points = points.cuda().to(device)
            mass=mass.cuda().to(device)
            evecs=evecs.cuda().to(device)
            evals=evals.cuda().to(device)
            gradx=gradx.cuda().to(device)
            grady=grady.cuda().to(device)

            basisNet = basisNet.eval()
            pred = basisNet(points,mass,lap,evals,evecs,gradx,grady)       
            basis_A = pred[1:,:,:]; basis_B = pred[:-1,:,:] 
            pc_A = points[1:,:,:]; pc_B = points[:-1,:,:]

            pseudo_inv_A = torch.pinverse(basis_A)
            C_opt = torch.matmul(pseudo_inv_A, basis_B)
            opt_A = torch.matmul(basis_A, C_opt)

            dist_matrix = torch.cdist(opt_A, basis_B)       
            s_max = torch.nn.Softmax(dim=1)
            s_max_matrix = s_max(-dist_matrix)
            eucl_loss = torch.sum(torch.square(torch.matmul(s_max_matrix, pc_B) - pc_B))
            eval_loss +=   eucl_loss.item()

        print('EPOCH ' + str(epoch) + ' - eva_loss: ' + str(eval_loss))

        # Saving if best model so far
        if eval_loss <  best_eval_loss:
            print('save model')
            best_eval_loss = eval_loss
            torch.save(basisNet.state_dict(), '%s/basis_model_best.pth' % (outf))

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)

        # Logging losses
        np.save(outf+'/train_losses_basis.npy',train_losses)
        np.save(outf+'/eval_losses_basis.npy',eval_losses)
     