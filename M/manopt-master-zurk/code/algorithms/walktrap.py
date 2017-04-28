import igraph

def walktrap(A, K):
    G = igraph.Graph.Adjacency((A > 0).tolist())
    clust = G.community_walktrap().as_clustering()
    return [[xt for xt in x] for x in clust]