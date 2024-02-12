import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy

def knn_query(X, Y, k=1, return_distance=False, n_jobs=1):
    tree = NearestNeighbors(n_neighbors=k, leaf_size=40, algorithm="kd_tree", n_jobs=n_jobs)
    tree.fit(X)
    dists, matches = tree.kneighbors(Y)

    if k == 1:
        dists = dists.squeeze()
        matches = matches.squeeze()

    if return_distance:
        return dists, matches
    return matches

def map_fit(basis1,basis2,desc1, desc2):
    F= np.matmul(np.linalg.pinv(basis1),desc1)
    G= np.matmul(np.linalg.pinv(basis2),desc2)
    return  np.linalg.lstsq(F.T,G.T,rcond=None)[0].T  #questa è la mappa che va da 1 a 2

#la mappa C è da considerarsi sempre in direzione 1-->2 
def FM_to_p2p(fmap12, basis1,basis2,adj=False,bijective=False): 
    if adj:
        emb2=np.matmul(basis2,fmap12)
        emb1=basis1
    else:
        emb1=np.matmul(basis1,fmap12.T)
        emb2=basis2
    if bijective:
        p2p=knn_query(emb2,emb1)        #p2p12
    else:
        p2p=knn_query(emb1,emb2)        #p2p21
    return p2p   


def p2p_to_FM(p2p,basis1,basis2, bijective=False):
    #if bijective is true p2p goes from 1 to 2, else it goes from 2 to 1
    # Pulled back eigenvectors
    if bijective:
        basis2_pb = basis2[p2p, :]
        return scipy.linalg.lstsq(basis2_pb,basis1)[0]
    else:
        basis1_pb = basis1[p2p, :]
        return scipy.linalg.lstsq(basis2,basis1_pb)[0]

    # Solve with least square
    #return scipy.linalg.lstsq(basis2,evects1_pb)[0]#scipy.linalg.lstsq(evects1_pb,basis2)[0].T # (k2,k1)

