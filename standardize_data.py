# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

import os.path as osp
import sys
import math
import numpy as np
import networkx as nx
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string("data", None, "")
flags.mark_flag_as_required("data")
FLAGS(sys.argv)

if FLAGS.data == "BlogCatalog":
    from datasets.BlogCatalog.standardize import standardize
elif FLAGS.data == "PPI":
    from datasets.PPI.standardize import standardize
elif FLAGS.data == "Mashup":
    from datasets.Mashup.standardize import standardize
elif FLAGS.data == "Flickr":
    from datasets.Flickr.standardize import standardize
else:
    raise ValueError
dataset_path = osp.join(
    osp.dirname(osp.abspath(__file__)), "datasets", FLAGS.data)

G = standardize()

num_nodes = G.number_of_nodes()
adj = nx.convert_matrix.to_scipy_sparse_matrix(
    G, nodelist=list(range(num_nodes)), format="csr")
degrees = np.asarray(adj.sum(axis=1), dtype=np.int64).flatten()

sampling_table_file = osp.join(dataset_path, "sampling_table.npy")
if osp.isfile(sampling_table_file):
    print("Sampling table already exists.")
else:
    table_size = 1e8
    power = 0.75
    numNodes = num_nodes

    print("Pre-procesing for non-uniform negative sampling!")
    node_degree = degrees.copy()
    # node_degree = np.zeros(numNodes)  # out degree

    # import ipdb; ipdb.set_trace()
    # look_up = self.g.look_up_dict
    # for edge in self.g.G.edges():
    #     node_degree[look_up[edge[0]]
    #                 ] += self.g.G[edge[0]][edge[1]]["weight"]

    norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

    sampling_table = np.zeros(int(table_size), dtype=np.uint32)

    p = 0
    i = 0
    for j in range(numNodes):
        p += float(math.pow(node_degree[j], power)) / norm
        while i < table_size and float(i) / table_size < p:
            sampling_table[i] = j
            i += 1
    np.save(sampling_table_file, sampling_table)
