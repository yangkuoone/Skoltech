import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from pyclustering.cluster.kmedians import kmedians
from scipy.sparse.linalg import eigsh
from big_clam import BigClam

eigs = sp.sparse.linalg.eigs

#TODO: split to functions, extract mutial with SAAC
def OCCAM(A, k, tau=None, clustering_mode='kmeans', W_mode='svd', norm_mode='l2', verbose=0):
    n = A.shape[0]
    if W_mode.lower() == 'svd':
        Lambda, U = eigs(1.0 * A, k, which="LR")
        Lambda, U = np.real(Lambda), np.real(U)
        if np.any(Lambda < 0):
            # From paper:  in practice some of the eigenvalues of A may
            #              be negative; if that happens, we truncate them to 0
            U, Lambda = U[:, Lambda > 0], Lambda[Lambda > 0]
            k = np.count_nonzero(Lambda > 0)
            print('Some of Lambda < 0. Now I just rewmove it, new k={}'.format(k))
        W = U * np.sqrt(Lambda)[:, None].T
    elif W_mode.lower() == 'bigclam':
        W, _ = BigClam(1.0 * (np.array(A.todense()) != 0), k, processesNo=1, LLH_output=False, stepSizeMod='simple',
                       initF='rand').fit()
    elif W_mode.lower() == 'nmf':
        W = NMF(n_components=k).fit_transform(A)
    else:
        raise ValueError('Unknown W_mode: ' + W_mode)

    if tau is None:
        hatAlpha = 1.0 * A.sum() / (n * (n - 1) * k)
        tau = 0.1 * hatAlpha ** 0.2 * k ** 1.5 / (n ** 0.3)
    if norm_mode == 'l2':
        hatW = W / (np.linalg.norm(W, axis=1) + tau)[:, None]
        if clustering_mode == 'kmeans':
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1231)
            kmeans.fit(hatW)
            hatS = kmeans.cluster_centers_
        elif clustering_mode == 'kmedians':
            cc = hatW[np.random.choice(hatW.shape[0], k)]
            km = kmedians(hatW, initial_centers=cc)
            km.process()
            hatS = np.array(km.get_medians())
        else:
            raise ValueError('Unknown clustering_mode: ' + clustering_mode)
    if norm_mode == 'l1':
        hatW = np.absolute(W) / (np.linalg.norm(W, ord=1, axis=1) + tau)[:, None]
        hatWcl = hatW[:, 1:]
        if clustering_mode == 'kmeans':
            kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1231)
            kmeans.fit(hatWcl)
            hatS = kmeans.cluster_centers_
        elif clustering_mode == 'kmedians':
            cc = hatW[np.random.choice(hatW.shape[0], k)]
            km = kmedians(hatW, initial_centers=cc)
            km.process()
            hatS = np.array(km.get_medians())
        else:
            raise ValueError('Unknown clustering_mode: ' + clustering_mode)
        hatS = np.concatenate(((1 - hatS.sum(axis=1))[:, None], hatS), axis=1)
        # print hatS

    # Project the rows of hatW onto hatS to obtain the coefficients
    # TODO: check formula
    projector = hatS.T.dot(np.linalg.inv(hatS.dot(hatS.T)))  # .dot(hatS))
    # projector = np.linalg.inv(hatS)
    hatTheta = hatW.dot(projector)
    hatTheta = hatTheta / np.linalg.norm(hatTheta, axis=1)[:, None]
    comm_matr = hatTheta >= 1.0 / k
    nodes, C = np.where(comm_matr)
    comms = [nodes[C == i] for i in range(k)]

    if verbose == 1:
        info = {}
        info['hatW'] = hatW
        info['hatS'] = hatS
        info['hatTheta'] = hatTheta
        info['hatTheta'] = hatTheta
        info['W'] = W
        info["tau"] = tau
        info["projector"] = projector
        info["comm_matr"] = comm_matr
        if W_mode.lower() == 'svd':
            info['Lambda'] = Lambda
            info['U'] = U

    return (comms, info) if verbose == 1 else comms
