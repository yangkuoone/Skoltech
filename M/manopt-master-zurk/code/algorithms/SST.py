# http://www.cc.gatech.edu/~mihail/D.8802readings/mcsherrystoc01.pdf

import numpy as np
import scipy.linalg
import networkx as nx
import scipy as sp
from math import floor
import random
from algorithms import NMI
from sklearn.cluster import KMeans

eigs = sp.sparse.linalg.eigs

def generate_graph(n, frac, p, q, **kwargs):

    if ("random_state" in kwargs):
        seed = kwargs["random_state"]
        np.random.seed(seed = seed)

    s1 = int(n*frac)
    s2 = n - s1
    g_in1 = p * np.ones((s1, s1))
    g_in2 = p * np.ones((s2, s2))
    g_out1 = q * np.ones((s1, s2))
    g_out2 = q * np.ones((s2, s1))

    block_model = np.bmat([[g_in1, g_out1], [g_out2, g_in2]])
    A = 1.0 * (np.random.rand(n, n) < block_model)
    true_comm = np.concatenate([ np.ones((s1, 1)), -np.ones((s2, 1))]).T

    cluster_1 = []
    cluster_2 = []
    for arg, node in enumerate(true_comm[0]):
        if node == 1:
            cluster_1.append(arg)
        else:
            cluster_2.append(arg)
    true_comm = [cluster_1, cluster_2]
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


def cproj(matrix, s_m=5, tau=1.5, k=2):

    A = matrix.T
    n_nodes = A.shape[1]
    unclassified_nodes = set(np.arange(n_nodes))
    T = []

    while len(unclassified_nodes) > s_m/2:
        v_i = np.random.choice(list(unclassified_nodes))
        T_i = []
        T_i.append(v_i)
        unclassified_nodes.remove(v_i)
        subtraction_matrix = A[:,v_i].reshape((A.shape[0],1)) - A
        multiplication_matrix = projection(matrix = A, x = subtraction_matrix, k=k)
        norms = np.linalg.norm(multiplication_matrix, axis=0)

        for u in range(n_nodes):
            if (u in unclassified_nodes) and (norms[u] <= tau):
                T_i.append(u)
                unclassified_nodes.remove(u)
        T.append(T_i)

    INF = float("INF")
    for u in unclassified_nodes:
        min_distance = INF
        tmp_u = 0
        for (T_i, i) in zip(T, range(len(T))):
            norm = np.linalg.norm(projection(matrix = A, x = A[:,T_i[0]] - A[:,u], k=k))
            if  norm < min_distance:
                min_distance = norm
                tmp_u = i
        T[tmp_u].append(u)

    C = []
    for T_i in T:
        C_i = np.arange(0, n_nodes)
        for node in range(n_nodes):
            if node in set(T_i):
                C_i[node] = 1
            else:
                C_i[node] = 0
        C.append(C_i)
    C = np.matrix(C)
    return C.T

def Clustering(graph, tau=1.5, k=2):

    if type(graph) == nx.classes.graph.Graph:
        hat_G = nx.adjacency_matrix(G = graph).toarray()
        n_nodes = hat_G.shape[0]
    else:
        hat_G = graph
        n_nodes = graph.shape[0]

    s_m = n_nodes/2
    permutation = np.random.permutation(n_nodes)
    first_part_of_nodes = permutation[: int(n_nodes/2)]
    second_part_of_nodes = permutation[int(n_nodes/2) :]

    A = hat_G[:,first_part_of_nodes]
    B = hat_G[:,second_part_of_nodes]

    P1 = projection(matrix=cproj(B, s_m=s_m, tau=tau), x=A, k=k)
    P2 = projection(matrix=cproj(A, s_m=s_m, tau=tau), x=B, k=k)
    hat_H = np.hstack((P1, P2))
    hat_H = np.zeros_like(hat_H)
    hat_H[:, first_part_of_nodes] = P1
    hat_H[:, second_part_of_nodes] = P2

    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1231)
    ans = np.array(kmeans.fit_predict(hat_H))

    T = []
    for i in range(ans.min(), ans.max()+1):
        T_i = []
        for ind, j in enumerate(ans):
            if j==i:
                T_i.append(ind)
        T.append(T_i)
    return T
