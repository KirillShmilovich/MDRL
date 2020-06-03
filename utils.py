from graphlets.utils import pbc
import networkx as nx
import numpy as np
from sklearn.neighbors import BallTree, radius_neighbors_graph


def compute_graph(X, dims, r_cut, metric=pbc):
    BT = BallTree(X, metric=metric, dims=dims)
    rng_con = radius_neighbors_graph(BT, r_cut, mode="connectivity")
    A = np.matrix(rng_con.toarray())
    G = nx.from_numpy_matrix(A)
    return G


def get_n_clusters(xyz, dims, r_cut, metric=pbc):
    G = compute_graph(xyz, dims, r_cut, metric)
    n_clusters = len(sorted(nx.connected_components(G)))
    return n_clusters
