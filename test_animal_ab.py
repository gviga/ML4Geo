#IMPORT LIBRARIES

#Libraries
import numpy as np
import torch
import hdf5storage

import sys
sys.path.append('./')


from utils.matching_fun import *
from utils.data_preprocess import *
from modelpp import PointNetBasis as PointNet2Basis
from modelpp import PointNetDesc as PointNet2Desc

from model import PointNetBasis,PointNetDesc 
from deltaconvbasis import DeltaConvBasis,DeltaConvDesc


from diffusionnet import DiffusionNetBasis,DiffusionNetDesc

from diffusion_net import geometry
from utils.metrics import *
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import potpourri3d as pp3d



def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    #m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    #pc = pc / m
    return pc



def ab_zo(p2p_21, evects1, evects2,k_init=20, nit=10, step=1):

    fmap_zo = p2p_to_FM(p2p_21, evects1[:,:k_init], evects2[:,:k_init],bijective=True) #C_{YX}, A_{XY}^T

    for l in range(k_init,k_init+nit+1):
        p2p_zo = FM_to_p2p(fmap_zo, evects1[:,:l], evects2[:,:l],adj=True,bijective=True) #T_{YX}
        fmap_zo = p2p_to_FM(p2p_zo, evects1[:,:l+1], evects2[:,:l+1],bijective=True) 

    return p2p_zo


v=np.load('./data/animal_val_easy2.npy' ).astype(np.float32)
#loading geodesic distance matrix (for evaluation)
#geod_load = hdf5storage.loadmat('./data/FAUST1k/N_out.mat')
#D = geod_load['D'].astype(np.float32)
#
solver=pp3d.PointCloudHeatSolver(v[0])
A_geod=np.zeros((1000,1000))
for i in range(v[0].shape[0]):
    A_geod[i,:]=solver.compute_distance(i)

p2p_gt = np.arange(1000)
n_esp=100


#landmarks
lm=[25,234,527,777,984]

k=30

bijective=True
adj=True
#Load data-driven basis


#POINTNET
pointnet_model = PointNetBasis(k=k, feature_transform=False)
checkpoint = torch.load('./models/pointnet/basis_model_best_animal.pth',map_location=torch.device('cuda'))
pointnet_model.load_state_dict(checkpoint)
pointnet_model = pointnet_model.eval()
pointnet_model.cuda()

pointnet_model_desc = PointNetDesc(k=40, feature_transform=False)
checkpoint = torch.load('./models/pointnet/desc_model_best_animal.pth',map_location=torch.device('cuda'))
pointnet_model_desc.load_state_dict(checkpoint)
pointnet_model_desc = pointnet_model_desc.eval()
pointnet_model_desc.cuda()

#POINTNET++

pointnet2_model = PointNet2Basis(k=k, feature_transform=False)
checkpoint = torch.load('./models/pointnetpp/basis_model_best_animal.pth',map_location=torch.device('cuda'))
pointnet2_model.load_state_dict(checkpoint)
pointnet2_model = pointnet2_model.eval()
pointnet2_model.cuda()

pointnet2_model_desc = PointNet2Desc(k=40, feature_transform=False)
checkpoint = torch.load('./models/pointnetpp/desc_model_best_animal.pth',map_location=torch.device('cuda'))
pointnet2_model_desc.load_state_dict(checkpoint)
pointnet2_model_desc = pointnet2_model_desc.eval()
pointnet2_model_desc.cuda()

'''
pointnet2_model_desc_var = PointNet2Desc(k=40, feature_transform=False)
checkpoint = torch.load('./models/pointnetpp/desc_model_best_animal.pth',map_location=torch.device('cuda'))
pointnet2_model_desc_var.load_state_dict(checkpoint)
pointnet2_model_desc_var = pointnet2_model_desc_var.eval()
pointnet2_model_desc_var.cuda()

'''
#DALTACONV
deltaconv_model = DeltaConvBasis(in_channels=3, k=30,  conv_channels=[64, 128, 256], mlp_depth=2, embedding_size=512, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1)
checkpoint = torch.load('./models/deltaconv/basis_model_best_animal.pth',map_location=torch.device('cuda'))
deltaconv_model.load_state_dict(checkpoint)
deltaconv_model=deltaconv_model.eval()

deltaconv_model.cuda()
deltaconv_model_desc = DeltaConvDesc(in_channels=3, k=40,  conv_channels=[64, 128, 256], mlp_depth=2, embedding_size=512, num_neighbors=20, grad_regularizer=0.001, grad_kernel_width=1)
checkpoint = torch.load('./models/deltaconv/desc_model_best_animal.pth',map_location=torch.device('cuda'))
deltaconv_model_desc.load_state_dict(checkpoint)
deltaconv_model_desc=deltaconv_model_desc.eval()
deltaconv_model_desc.cuda()

#DIFFUSIONNET

diffusionnet_model =DiffusionNetBasis(k=30,n_block=12, feature_transform=False)
checkpoint = torch.load('./models/diffusionnet/basis_model_best_animal.pth',map_location=torch.device('cuda'))
diffusionnet_model.load_state_dict(checkpoint)
diffusionnet_model=diffusionnet_model.eval()
diffusionnet_model.cuda()

diffusionnet_model_desc = DiffusionNetDesc(k=40,n_block=12, feature_transform=False)

checkpoint = torch.load('./models/diffusionnet/desc_model_best_animal.pth',map_location=torch.device('cuda'))
diffusionnet_model_desc.load_state_dict(checkpoint)
diffusionnet_model_desc=diffusionnet_model_desc.eval()
diffusionnet_model_desc.cuda()

match_lie=np.zeros((n_esp,p2p_gt.shape[0]))
err_lie_geod=np.zeros((n_esp,p2p_gt.shape[0]))
err_lie_eu=np.zeros((n_esp,p2p_gt.shape[0]))
ortho_lie=np.zeros((n_esp,))
bij_lie=np.zeros((n_esp,))
inj_lie=np.zeros((n_esp,))
conversion_err_lie=np.zeros((n_esp,p2p_gt.shape[0]))


match_lie_B=np.zeros((n_esp,p2p_gt.shape[0]))
err_lie_geod_B=np.zeros((n_esp,p2p_gt.shape[0]))
err_lie_eu_B=np.zeros((n_esp,p2p_gt.shape[0]))
ortho_lie_B=np.zeros((n_esp,))
bij_lie_B=np.zeros((n_esp,))
inj_lie_B=np.zeros((n_esp,))
conversion_err_lie_B=np.zeros((n_esp,p2p_gt.shape[0]))
#TEST

print('POINTNET:\n')


np.random.seed(2)
for i in range(n_esp):
    #shape selection
    vec = np.random.randint(300,size=2)
    if bijective:
        shapen1 = vec[1]
        shapen2 = vec[0]
    else:
        shapen1 = vec[0]
        shapen2 = vec[1]
    #save shapes
    v1 = pc_normalize(v[shapen1,:,:].squeeze())
    v2 = pc_normalize(v[shapen2,:,:].squeeze())

    solver=pp3d.PointCloudHeatSolver(v2)
    A_geod=np.zeros((1000,1000))
    for ll in range(v2.shape[0]):
        A_geod[ll,:]=solver.compute_distance(ll)
       # Computing Basis and Descriptors
    pred_basis = pointnet_model(torch.transpose(torch.tensor(v[[shapen1,shapen2],:,:].astype(np.float32),device='cuda'),1,2))
    basis = pred_basis[0].cpu().detach().numpy()
    pred_desc = pointnet_model_desc(torch.transpose(torch.tensor(v[[shapen1,shapen2],:,:].astype(np.float32),device='cuda'),1,2))
    desc = pred_desc[0].cpu().detach().numpy()
    # Saving basis amd descriptors
    basis1 = np.squeeze(basis[0])
    basis2 = np.squeeze(basis[1])
    desc1 = np.squeeze(desc[0])
    desc2 = np.squeeze(desc[1])

    #map optimization
    fmap = map_fit(basis1[:,:20],basis2[:,:20],desc1,desc2)   #from v1 to v2

    #map conversion (if bijective p2p from v1 to v2), (else p2p from v2 to v1)
    p2p = FM_to_p2p(fmap, basis1[:,:20], basis2[:,:20],adj, bijective)
    match_lie[i,:]=p2p
    #evaluation before refinement
    err_lie_geod_B[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu_B[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu_B[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie_B[i]=eval_bijectivity(p2p,p2p_gt)
    inj_lie_B[i]=eval_injectivity2(p2p,p2p_gt)

    ortho_lie_B[i]=eval_orthogonality(fmap)
    p2p=ab_zo(p2p,basis1,basis2,20,10)
    #evaluation module
    err_lie_geod[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie[i]=eval_bijectivity(p2p,p2p_gt)
    inj_lie[i]=eval_injectivity2(p2p,p2p_gt)

    ortho_lie[i]=eval_orthogonality(fmap)
    
    ortho_lie[i]=eval_orthogonality(fmap)
    fmap_gt=p2p_to_FM(p2p_gt, basis1, basis2,bijective)
    p2p_conv=FM_to_p2p(fmap_gt,basis1,basis2,adj,bijective)
    conversion_err_lie[i,:]=A_geod[(p2p_conv, p2p_gt)]

print('INI\n')

print(f'Errore medio geodesico Lie: {np.mean(np.mean(err_lie_geod_B,1),0):4f} \n'
      f'Errore medio euclideo Lie: {np.mean(np.mean(err_lie_eu_B,1),0):4f} \n'
      f'Errore biettività Lie: {np.mean(bij_lie_B):4f} \n'
      f'Errore iniettività Lie: {np.mean(inj_lie_B):4f} \n'
      f'Errore orthogonalità Lie: {np.mean(ortho_lie_B):4f} \n'
      f'Errore Conversione Lie : {np.mean(np.mean(conversion_err_lie_B,1),0):4f} \n')

print('REF\n')
print(f'Errore medio geodesico Lie: {np.mean(np.mean(err_lie_geod,1),0):4f} \n'
      f'Errore medio euclideo Lie: {np.mean(np.mean(err_lie_eu,1),0):4f} \n'
      f'Errore biettività Lie: {np.mean(bij_lie):4f} \n'
      f'Errore iniettività Lie: {np.mean(inj_lie):4f} \n'
      f'Errore orthogonalità Lie: {np.mean(ortho_lie):4f} \n'
      f'Errore Conversione Lie : {np.mean(np.mean(conversion_err_lie,1),0):4f} \n')
    

print('POINTNET++')

np.random.seed(2)
for i in range(n_esp):
    #shape selection
    vec = np.random.randint(300,size=2)
    if bijective:
        shapen1 = vec[1]
        shapen2 = vec[0]
    else:
        shapen1 = vec[0]
        shapen2 = vec[1]
    #save shapes
    v1 = pc_normalize(v[shapen1,:,:].squeeze())
    v2 = pc_normalize(v[shapen2,:,:].squeeze())

    solver=pp3d.PointCloudHeatSolver(v2)
    A_geod=np.zeros((1000,1000))
    for ll in range(v2.shape[0]):
        A_geod[ll,:]=solver.compute_distance(ll)

       # Computing Basis and Descriptors
    # Computing Basis and Descriptors
    pred_basis = pointnet2_model(torch.transpose(torch.tensor(v[[shapen1,shapen2],:,:].astype(np.float32),device='cuda'),1,2))
    basis = pred_basis.cpu().detach().numpy()
    pred_desc = pointnet2_model_desc(torch.transpose(torch.tensor(v[[shapen1,shapen2],:,:].astype(np.float32),device='cuda'),1,2))
    desc = pred_desc.cpu().detach().numpy()
    # Saving basis amd descriptors
    basis1 = np.squeeze(basis[0])
    basis2 = np.squeeze(basis[1])
    desc1 = np.squeeze(desc[0])
    desc2 = np.squeeze(desc[1])


    #map optimization
    fmap = map_fit(basis1[:,:20],basis2[:,:20],desc1,desc2)   #from v1 to v2

    #map conversion (if bijective p2p from v1 to v2), (else p2p from v2 to v1)
    p2p = FM_to_p2p(fmap, basis1[:,:20], basis2[:,:20],adj, bijective)
    match_lie[i,:]=p2p
    #evaluation before refinement
    err_lie_geod_B[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu_B[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu_B[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie_B[i]=eval_bijectivity(p2p,p2p_gt)
    inj_lie_B[i]=eval_injectivity2(p2p,p2p_gt)

    ortho_lie_B[i]=eval_orthogonality(fmap)
    p2p=ab_zo(p2p,basis1,basis2,20,10)
    #evaluation module
    err_lie_geod[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie[i]=eval_bijectivity(p2p,p2p_gt)
    inj_lie[i]=eval_injectivity2(p2p,p2p_gt)

    ortho_lie[i]=eval_orthogonality(fmap)
    
    ortho_lie[i]=eval_orthogonality(fmap)
    fmap_gt=p2p_to_FM(p2p_gt, basis1, basis2,bijective)
    p2p_conv=FM_to_p2p(fmap_gt,basis1,basis2,adj,bijective)
    conversion_err_lie[i,:]=A_geod[(p2p_conv, p2p_gt)]

print('INI\n')

print(f'Errore medio geodesico Lie: {np.mean(np.mean(err_lie_geod_B,1),0):4f} \n'
      f'Errore medio euclideo Lie: {np.mean(np.mean(err_lie_eu_B,1),0):4f} \n'
      f'Errore biettività Lie: {np.mean(bij_lie_B):4f} \n'
      f'Errore iniettività Lie: {np.mean(inj_lie_B):4f} \n'
      f'Errore orthogonalità Lie: {np.mean(ortho_lie_B):4f} \n'
      f'Errore Conversione Lie : {np.mean(np.mean(conversion_err_lie_B,1),0):4f} \n')

print('REF\n')
print(f'Errore medio geodesico Lie: {np.mean(np.mean(err_lie_geod,1),0):4f} \n'
      f'Errore medio euclideo Lie: {np.mean(np.mean(err_lie_eu,1),0):4f} \n'
      f'Errore biettività Lie: {np.mean(bij_lie):4f} \n'
      f'Errore iniettività Lie: {np.mean(inj_lie):4f} \n'
      f'Errore orthogonalità Lie: {np.mean(ortho_lie):4f} \n'
      f'Errore Conversione Lie : {np.mean(np.mean(conversion_err_lie,1),0):4f} \n')
    
'''
print('POINTNET BASIS + POINET++ DESCRIPTORS')


np.random.seed(2)
for i in range(n_esp):
    #shape selection
    #vec = np.random.randint(100,size=2)
    if bijective:
        shapen1 = 1
        shapen2 = 0
    else:
        shapen1 = 0
        shapen2 = 1
    #save shapes
    v1 = pc_normalize(v[shapen1,:,:].squeeze())
    v2 = pc_normalize(v[shapen2,:,:].squeeze())

    solver=pp3d.PointCloudHeatSolver(v2)
    A_geod=np.zeros((1000,1000))
    for ll in range(v2.shape[0]):
        A_geod[ll,:]=solver.compute_distance(ll)

       # Computing Basis and Descriptors
    # Computing Basis and Descriptors
    pred_basis = pointnet_model(torch.transpose(torch.tensor(v[[shapen1,shapen2],:,:].astype(np.float32),device='cuda'),1,2))
    basis = pred_basis[0].cpu().detach().numpy()
    pred_desc = pointnet2_model_desc_var(torch.transpose(torch.tensor(v[[shapen1,shapen2],:,:].astype(np.float32),device='cuda'),1,2))
    desc = pred_desc.cpu().detach().numpy()
    # Saving basis amd descriptors
    basis1 = np.squeeze(basis[0])
    basis2 = np.squeeze(basis[1])
    desc1 = np.squeeze(desc[0])
    desc2 = np.squeeze(desc[1])
    #map optimization
    fmap = map_fit(basis1[:,:20],basis2[:,:20],desc1,desc2)   #from v1 to v2

    #map conversion (if bijective p2p from v1 to v2), (else p2p from v2 to v1)
    p2p = FM_to_p2p(fmap, basis1[:,:20], basis2[:,:20],adj, bijective)
    match_lie[i,:]=p2p
    #evaluation before refinement
    err_lie_geod_B[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu_B[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu_B[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie_B[i]=eval_bijectivity(p2p,p2p_gt)
    inj_lie_B[i]=eval_injectivity2(p2p,p2p_gt)

    ortho_lie_B[i]=eval_orthogonality(fmap)
    p2p=ab_zo(p2p,basis1,basis2,20,10)
    #evaluation module
    err_lie_geod[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie[i]=eval_bijectivity(p2p,p2p_gt)
    inj_lie[i]=eval_injectivity2(p2p,p2p_gt)

    ortho_lie[i]=eval_orthogonality(fmap)
    
    ortho_lie[i]=eval_orthogonality(fmap)
    fmap_gt=p2p_to_FM(p2p_gt, basis1, basis2,bijective)
    p2p_conv=FM_to_p2p(fmap_gt,basis1,basis2,adj,bijective)
    conversion_err_lie[i,:]=A_geod[(p2p_conv, p2p_gt)]

print('INI\n')

print(f'Errore medio geodesico Lie: {np.mean(np.mean(err_lie_geod_B,1),0):4f} \n'
      f'Errore medio euclideo Lie: {np.mean(np.mean(err_lie_eu_B,1),0):4f} \n'
      f'Errore biettività Lie: {np.mean(bij_lie_B):4f} \n'
      f'Errore iniettività Lie: {np.mean(inj_lie_B):4f} \n'
      f'Errore orthogonalità Lie: {np.mean(ortho_lie_B):4f} \n'
      f'Errore Conversione Lie : {np.mean(np.mean(conversion_err_lie_B,1),0):4f} \n')

print('REF\n')
print(f'Errore medio geodesico Lie: {np.mean(np.mean(err_lie_geod,1),0):4f} \n'
      f'Errore medio euclideo Lie: {np.mean(np.mean(err_lie_eu,1),0):4f} \n'
      f'Errore biettività Lie: {np.mean(bij_lie):4f} \n'
      f'Errore iniettività Lie: {np.mean(inj_lie):4f} \n'
      f'Errore orthogonalità Lie: {np.mean(ortho_lie):4f} \n'
      f'Errore Conversione Lie : {np.mean(np.mean(conversion_err_lie,1),0):4f} \n')
    
'''

print('DELTACONV:')

np.random.seed(2)
for i in range(n_esp):
    #shape selection
    vec = np.random.randint(100,size=2)
    if bijective:
        shapen1 = vec[1]
        shapen2 = vec[0]
    else:
        shapen1 = vec[0]
        shapen2 = vec[1]
    #save shapes
    v1 = pc_normalize(v[shapen1,:,:].squeeze())
    v2 = pc_normalize(v[shapen2,:,:].squeeze())

    solver=pp3d.PointCloudHeatSolver(v2)
    A_geod=np.zeros((1000,1000))
    for ll in range(v2.shape[0]):
        A_geod[ll,:]=solver.compute_distance(ll)

       # Computing Basis and Descriptors
    pred_basis1 = deltaconv_model(Data(pos=torch.tensor(v1,device='cuda'),batch=torch.zeros((1000,),dtype=torch.int64,device='cuda')))
    basis1 = pred_basis1.cpu().detach().numpy()
    basis1 = np.squeeze(basis1)
    pred_basis2 = deltaconv_model(Data(pos=torch.tensor(v2,device='cuda'),batch=torch.zeros((1000,),dtype=torch.int64,device='cuda')))
    basis2 = pred_basis2.cpu().detach().numpy()
    basis2 = np.squeeze(basis2)

    pred_desc1 = deltaconv_model_desc(Data(pos=torch.tensor(v1,device='cuda'),batch=torch.zeros((1000,),dtype=torch.int64,device='cuda')))
    desc1 = pred_desc1.cpu().detach().numpy()
    desc1 = np.squeeze(desc1)
    pred_desc2 = deltaconv_model_desc(Data(pos=torch.tensor(v2,device='cuda'),batch=torch.zeros((1000,),dtype=torch.int64,device='cuda')))
    desc2 = pred_desc2.cpu().detach().numpy()
    desc2 = np.squeeze(desc2)

    #map optimization
    fmap = map_fit(basis1[:,:20],basis2[:,:20],desc1,desc2)   #from v1 to v2

    #map conversion (if bijective p2p from v1 to v2), (else p2p from v2 to v1)
    p2p = FM_to_p2p(fmap, basis1[:,:20], basis2[:,:20],adj, bijective)
    match_lie[i,:]=p2p
    #evaluation before refinement
    err_lie_geod_B[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu_B[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu_B[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie_B[i]=eval_bijectivity(p2p,p2p_gt)
    inj_lie_B[i]=eval_injectivity2(p2p,p2p_gt)

    ortho_lie_B[i]=eval_orthogonality(fmap)
    p2p=ab_zo(p2p,basis1,basis2,20,10)
    #evaluation module
    err_lie_geod[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie[i]=eval_bijectivity(p2p,p2p_gt)
    inj_lie[i]=eval_injectivity2(p2p,p2p_gt)

    ortho_lie[i]=eval_orthogonality(fmap)
    
    ortho_lie[i]=eval_orthogonality(fmap)
    fmap_gt=p2p_to_FM(p2p_gt, basis1, basis2,bijective)
    p2p_conv=FM_to_p2p(fmap_gt,basis1,basis2,adj,bijective)
    conversion_err_lie[i,:]=A_geod[(p2p_conv, p2p_gt)]

print('INI\n')

print(f'Errore medio geodesico Lie: {np.mean(np.mean(err_lie_geod_B,1),0):4f} \n'
      f'Errore medio euclideo Lie: {np.mean(np.mean(err_lie_eu_B,1),0):4f} \n'
      f'Errore biettività Lie: {np.mean(bij_lie_B):4f} \n'
      f'Errore iniettività Lie: {np.mean(inj_lie_B):4f} \n'
      f'Errore orthogonalità Lie: {np.mean(ortho_lie_B):4f} \n'
      f'Errore Conversione Lie : {np.mean(np.mean(conversion_err_lie_B,1),0):4f} \n')

print('REF\n')
print(f'Errore medio geodesico Lie: {np.mean(np.mean(err_lie_geod,1),0):4f} \n'
      f'Errore medio euclideo Lie: {np.mean(np.mean(err_lie_eu,1),0):4f} \n'
      f'Errore biettività Lie: {np.mean(bij_lie):4f} \n'
      f'Errore iniettività Lie: {np.mean(inj_lie):4f} \n'
      f'Errore orthogonalità Lie: {np.mean(ortho_lie):4f} \n'
      f'Errore Conversione Lie : {np.mean(np.mean(conversion_err_lie,1),0):4f} \n')
    



print('DIFFUSIONNET:')

np.random.seed(2)
for i in range(n_esp):
    #shape selection
    vec = np.random.randint(100,size=2)
    if bijective:
        shapen1 = vec[1]
        shapen2 = vec[0]
    else:
        shapen1 = vec[0]
        shapen2 = vec[1]
    #save shapes
    v1 = pc_normalize(v[shapen1,:,:].squeeze())
    v2 = pc_normalize(v[shapen2,:,:].squeeze())

    solver=pp3d.PointCloudHeatSolver(v2)
    A_geod=np.zeros((1000,1000))
    for ll in range(v2.shape[0]):
        A_geod[ll,:]=solver.compute_distance(ll)

    #save shapes
    v1 = torch.tensor(v[shapen1,:,:])
    v2 = torch.tensor(v[shapen2,:,:])

    #compute laplace beltrami eigenfunctions
    face=torch.empty(0)
    frames1, mass1, L1, evals1, evecs1, gradX1, gradY1 = geometry.compute_operators(v1, face, 128)
    frames2, mass2, L2, evals2, evecs2, gradX2, gradY2 = geometry.compute_operators(v2, face,k_eig=128)
    #np.save('./out/basis1.npy',evecs[0][:,:basis_dim].detach())

    v1=v1.unsqueeze(0).cuda()
    mass1=mass1.unsqueeze(0).cuda()
    evecs1=evecs1.unsqueeze(0).cuda()
    L1=L1.unsqueeze(0).cuda()
    evals1=evals1.unsqueeze(0).cuda()
    gradX1=gradX1.unsqueeze(0).cuda()
    gradY1=gradY1.unsqueeze(0).cuda()

    v2=v2.unsqueeze(0).cuda()
    mass2=mass2.unsqueeze(0).cuda()
    L2=L2.unsqueeze(0).cuda()
    evecs2=evecs2.unsqueeze(0).cuda()
    evals2=evals2.unsqueeze(0).cuda()
    gradX2=gradX2.unsqueeze(0).cuda()
    gradY2=gradY2.unsqueeze(0).cuda()
    # Computing Basis and Descriptors
    pred_basis1 = diffusionnet_model(v1,mass1,L1,evals1,evecs1,gradX1,gradY1)
    basis1 = pred_basis1.cpu().detach().numpy()
    basis1 = np.squeeze(basis1)
    pred_basis2 = diffusionnet_model(v2,mass2,L2,evals2,evecs2,gradX2,gradY2)
    basis2 = pred_basis2.cpu().detach().numpy()
    basis2 = np.squeeze(basis2)
    
    pred_desc1 = diffusionnet_model_desc(v1,mass1,L1,evals1,evecs1,gradX1,gradY1)
    desc1 = pred_desc1.cpu().detach().numpy()
    desc1 = np.squeeze(desc1)
    pred_desc2 = diffusionnet_model_desc(v2,mass2,L2,evals2,evecs2,gradX2,gradY2)
    desc2 = pred_desc2.cpu().detach().numpy()
    desc2 = np.squeeze(desc2)
    
    v1=np.squeeze(v1.cpu().detach().numpy())
    v2=np.squeeze(v2.cpu().detach().numpy())
    #map optimization
    #map optimization
    fmap = map_fit(basis1[:,:20],basis2[:,:20],desc1,desc2)   #from v1 to v2

    #map conversion (if bijective p2p from v1 to v2), (else p2p from v2 to v1)
    p2p = FM_to_p2p(fmap, basis1[:,:20], basis2[:,:20],adj, bijective)
    match_lie[i,:]=p2p
    #evaluation before refinement
    err_lie_geod_B[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu_B[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu_B[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie_B[i]=eval_bijectivity(p2p,p2p_gt)
    inj_lie_B[i]=eval_injectivity2(p2p,p2p_gt)

    ortho_lie_B[i]=eval_orthogonality(fmap)
    p2p=ab_zo(p2p,basis1,basis2,20,10)
    #evaluation module
    err_lie_geod[i,:] =A_geod[(p2p, p2p_gt)]
    if bijective: 
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v2[p2p_gt.astype(np.int32)] - v2[p2p,:]),1))    #euclidean error
    else:
        err_lie_eu[i,:] = np.sqrt(np.sum(np.square(v1[p2p_gt.astype(np.int32)] - v1[p2p,:]),1))     #euclidean error

    bij_lie[i]=eval_bijectivity(p2p,p2p_gt)
    inj_lie[i]=eval_injectivity2(p2p,p2p_gt)

    ortho_lie[i]=eval_orthogonality(fmap)
    
    ortho_lie[i]=eval_orthogonality(fmap)
    fmap_gt=p2p_to_FM(p2p_gt, basis1, basis2,bijective)
    p2p_conv=FM_to_p2p(fmap_gt,basis1,basis2,adj,bijective)
    conversion_err_lie[i,:]=A_geod[(p2p_conv, p2p_gt)]

print('INI\n')

print(f'Errore medio geodesico Lie: {np.mean(np.mean(err_lie_geod_B,1),0):4f} \n'
      f'Errore medio euclideo Lie: {np.mean(np.mean(err_lie_eu_B,1),0):4f} \n'
      f'Errore biettività Lie: {np.mean(bij_lie_B):4f} \n'
      f'Errore iniettività Lie: {np.mean(inj_lie_B):4f} \n'
      f'Errore orthogonalità Lie: {np.mean(ortho_lie_B):4f} \n'
      f'Errore Conversione Lie : {np.mean(np.mean(conversion_err_lie_B,1),0):4f} \n')

print('REF\n')
print(f'Errore medio geodesico Lie: {np.mean(np.mean(err_lie_geod,1),0):4f} \n'
      f'Errore medio euclideo Lie: {np.mean(np.mean(err_lie_eu,1),0):4f} \n'
      f'Errore biettività Lie: {np.mean(bij_lie):4f} \n'
      f'Errore iniettività Lie: {np.mean(inj_lie):4f} \n'
      f'Errore orthogonalità Lie: {np.mean(ortho_lie):4f} \n'
      f'Errore Conversione Lie : {np.mean(np.mean(conversion_err_lie,1),0):4f} \n')
    