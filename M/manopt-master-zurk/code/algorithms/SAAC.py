import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from pyclustering.cluster.kmedians import kmedians
from scipy.sparse.linalg import eigsh
from big_clam import BigClam

eigs = sp.sparse.linalg.eigs


from itertools import combinations

#TODO: split to functions, extract mutial with OCCAM
def SAAC(A, k, s=2, eps=1e-4, clustering_mode='kmeans', W_mode='svd', verbose=0):

    if W_mode.lower() == 'svd':
        Lambda, U = eigs(1.0 * A, k, which="LR")
        Lambda, U = np.real(Lambda), np.real(U)
        if np.any(Lambda < 0):
            # From paper:  in practice some of the eigenvalues of A may
            #              be negative; if that happens, we truncate them to 0
            U, Lambda = U[:, Lambda > 0], Lambda[Lambda > 0]
            k = np.count_nonzero(Lambda > 0)
            print('Some of Lambda < 0. Now I just rewmove it, new k={}'.format(k))
        W = U# * np.sqrt(Lambda)[:,None].T
    elif W_mode.lower() == 'bigclam':
        W, _ = BigClam(1.0 * (np.array(A.todense()) != 0), k, processesNo=1, LLH_output=False, stepSizeMod='simple', initF='rand').fit()
    elif W_mode.lower() == 'nmf':
        W = NMF(n_components=k).fit_transform(A)
    else:
        raise ValueError('Unknown W_mode: ' + W_mode)


    if clustering_mode == 'kmeans':
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1231)
        kmeans.fit(W)
        hatX = kmeans.cluster_centers_
    elif clustering_mode == 'kmedians':
        km = kmedians(W, initial_centers=W[np.random.choice(W.shape[0], k)])
        hatX = np.array(km.get_medians())
    else:
        raise ValueError('Unknown clustering_mode: ' + clustering_mode)

    loss = float('inf')
    hatX_init = hatX.copy()
    hatZ = np.zeros((A.shape[0], k))
    while loss - np.linalg.norm(W-hatZ.dot(hatX), ord='fro') > eps:
        loss = np.linalg.norm(W-hatZ.dot(hatX), ord='fro')
        #update membership vectors
        for i in range(hatZ.shape[0]):
            best_ones_indx = None
            best_dist = float('inf')
            for over in range(s+1):
                for ones_indx in combinations(range(k), over):
                    zhatX = np.sum(hatX[ones_indx,:], axis=0)
                    dist = np.linalg.norm(W[i,:] - zhatX)
                    if dist < best_dist:
                        best_dist = dist
                        best_ones_indx = ones_indx
            hatZ[i, :] = 0
            hatZ[i, best_ones_indx] = 1

        hatX = np.linalg.inv(hatZ.T.dot(hatZ)).dot(hatZ.T.dot(W))

    comm_matr = hatZ == 1
    nodes, C = np.where(comm_matr)
    comms = [nodes[C==i] for i in range(k)]

    if verbose == 1:
        info = {}
        info['hatX'] = hatX
        info['hatX_init'] = hatX_init
        info['hatZ'] = hatZ
        info['W'] = W
        if W_mode.lower() == 'svd':
            info['Lambda'] = Lambda
            info['W'] = W

    return (comms, info) if verbose == 1 else comms
