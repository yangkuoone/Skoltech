import numpy as np
from sklearn.decomposition import NMF


def NMF_clust(A, K):
    model = NMF(n_components=K)
    res = model.fit_transform(A)
    res = res / res.sum(axis=1)[:, None]
    nodes, C = np.where(res >= 1.0 / K)
    comms = [nodes[C == i] for i in range(K)]
    return comms
