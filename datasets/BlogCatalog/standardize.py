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
import networkx as nx
from collections import defaultdict


def standardize():
    filepath = osp.dirname(osp.abspath(__file__))

    # load nodes
    with open(f"{filepath}/original/nodes.csv", "r") as f:
        nodes = {line.rstrip() for line in f}
    node_mapping = {x: int(x) - 1 for x in nodes}

    # load edges
    with open(f"{filepath}/original/edges.csv", "r") as f:
        edges = {tuple(line.rstrip().split(",")) for line in f}

    # load groups
    with open(f"{filepath}/original/groups.csv", "r") as f:
        groups = {line.rstrip() for line in f}
    group_mapping = {x: int(x) - 1 for x in groups}

    # load node-group edges
    node_groups = defaultdict(set)
    with open(f"{filepath}/original/group-edges.csv", "r") as f:
        for line in f:
            node, group = line.rstrip().split(",")
            node_groups[node].add(group)
    node_groups = dict(node_groups)

    G = nx.Graph()
    # add nodes
    for node in nodes:
        G.add_node(node_mapping[node])
    # add edges
    for edge in edges:
        G.add_edge(node_mapping[edge[0]], node_mapping[edge[1]])
    # add groups
    for node, groups in node_groups.items():
        mapped_groups = [group_mapping[x] for x in groups]
        label = [int(i in mapped_groups) for i in range(len(group_mapping))]
        G.nodes[node_mapping[node]]["label"] = label

    label_array = np.asarray(
        [G.nodes[i]["label"] for i in range(G.number_of_nodes())],
        dtype=np.int64)
    G.graph["label_array"] = label_array

    nx.write_gpickle(G, f"{filepath}/BlogCatalog_nx.pkl")
    nx.write_gpickle(G, f"{filepath}/BlogCatalog_nx.pkl2", protocol=2)

    return G
