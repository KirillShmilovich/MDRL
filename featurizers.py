import numpy as np
from graphlets import Graphlets
from graphlets.utils import pbc


def graphlet_featurizer(xyz, dims, r_cut, metric=pbc):
    xyz = np.expand_dims(xyz, 0)
    dims = np.expand_dims(dims, 0)
    G = Graphlets(xyz, dims=dims, metric=metric)
    return G.compute(r_cut=r_cut).flatten()
