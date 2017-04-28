import numpy as np
import scipy as sp
import scipy.linalg
import networkx as nx
from math import floor
import random
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from pyclustering.cluster.kmedians import kmedians

eigs = sp.sparse.linalg.eigs


def generate_graph(n, frac, p, q, **kwargs):
    if "random_state" in kwargs:
        seed = kwargs["random_state"]
        np.random.seed(seed)

    s1 = int(n * frac)
    s2 = n - s1
    g_in1 = p * np.ones((s1, s1))
    g_in2 = p * np.ones((s2, s2))
    g_out1 = q * np.ones((s1, s2))
    g_out2 = q * np.ones((s2, s1))

    block_model = np.bmat([[g_in1, g_out1], [g_out2, g_in2]])
    A = 1.0 * (np.random.rand(n, n) < block_model)
    true_comm = np.concatenate([np.ones((s1, 1)), -np.ones((s2, 1))]).T

    cluster_1 = []
    cluster_2 = []
    for arg, node in enumerate(true_comm[0]):
        if node == 1:
            cluster_1.append(arg)
        else:
            cluster_2.append(arg)
    true_comm = [cluster_1, cluster_2]
    A = 1.0 * ((A.T + A) != 0)
    return block_model, A, true_comm


def projection(matrix, x, k):
    Lambda, U = eigs(1.0 * matrix.dot(matrix.T), k, which="LR")
    Lambda, U = np.real(Lambda), np.real(U)
    if np.any(Lambda < 0):
        # From paper:  in practice some of the eigenvalues of A may
        #              be negative; if that happens, we truncate them to 0
        U, Lambda = U[:, Lambda > 0], Lambda[Lambda > 0]
        k = np.count_nonzero(Lambda > 0)
        print('Some of Lambda < 0. Now I just remove it, new k={}'.format(k))

    return np.dot(np.dot(U, U.T), x)


def Clustering(graph, k=2, clustering_mode='kmeans', **kwargs):
    if type(graph) == nx.classes.graph.Graph:
        hat_G = nx.adjacency_matrix(G=graph).toarray()
        n_nodes = hat_G.shape[0]
    else:
        hat_G = graph
        n_nodes = graph.shape[0]

    permutation = np.random.permutation(n_nodes)
    first_part_of_nodes = permutation[: int(n_nodes / 2)]
    second_part_of_nodes = permutation[int(n_nodes / 2):]
    A = hat_G[:, first_part_of_nodes]
    B = hat_G[:, second_part_of_nodes]

    hat_H = np.zeros((n_nodes, n_nodes))
    hat_H[:, first_part_of_nodes] = projection(matrix=B, x=A, k=k)
    hat_H[:, second_part_of_nodes] = projection(matrix=A, x=B, k=k)

    if clustering_mode == 'kmeans':
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
        ans = np.array(kmeans.fit_predict(hat_H))
    elif clustering_mode == 'kmedians':
        cc = hat_H[np.random.choice(hat_H.shape[0], k)]
        km = kmedians(hat_H, initial_centers=cc)
        km.process()
        ans = np.array(km.get_clusters())
        return ans
    else:
        raise ValueError('Unknown clustering_mode: ' + clustering_mode)

    T = []
    for i in range(ans.min(), ans.max() + 1):
        T_i = []
        for ind, j in enumerate(ans):
            if j == i:
                T_i.append(ind)
        T.append(T_i)
    return T
