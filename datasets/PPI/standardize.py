# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

import os.path as osp
import numpy as np
import scipy as sp
import scipy.io
import networkx as nx


def standardize():
    filepath = osp.dirname(osp.abspath(__file__))

    data = sp.io.loadmat(f"{filepath}/Homo_sapiens.mat")
    network, group = data["network"], data["group"]

    network.setdiag(0.0)
    network.eliminate_zeros()
    degree = np.asarray(network.sum(1)).flatten()

    network = network[degree > 0][:, degree > 0]
    group = group[degree > 0]

    G = nx.Graph()
    # add nodes
    for i in range(network.shape[0]):
        G.add_node(i)
    # add edges
    for i, j in zip(*network.nonzero()):
        if i != j:
            G.add_edge(i, j)
            G.add_edge(j, i)
    # add groups
    for i, row in enumerate(group):
        label = row.toarray().flatten().astype(np.int32).tolist()
        G.nodes[i]["label"] = label

    label_array = np.asarray(
        [G.nodes[i]["label"] for i in range(G.number_of_nodes())],
        dtype=np.int64)
    G.graph["label_array"] = label_array

    nx.write_gpickle(G, f"{filepath}/PPI_nx.pkl")
    nx.write_gpickle(G, f"{filepath}/PPI_nx.pkl2", protocol=2)

    return G
