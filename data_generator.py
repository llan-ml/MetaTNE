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
import tensorflow as tf
from tensorflow.python.util import nest

from utils import sample_tasks, batching_tasks


class DataGenerator(object):
    def __init__(self, FLAGS, training=True):
        self.seed = FLAGS.seed
        self.rng_basic = np.random.RandomState(self.seed)
        self.rng_tasks = np.random.RandomState(self.seed)
        self.dataset = FLAGS.dataset_str
        self.dataset_path = osp.join(
            osp.dirname(osp.abspath(__file__)), "datasets",
            f"{FLAGS.dataset_str}/{FLAGS.dataset_str}_nx.pkl")
        self.unsup_batch_size = FLAGS.unsup_batch_size
        self.unsup_num_neg_samples = FLAGS.unsup_num_neg_samples
        self.meta_batch_size = FLAGS.meta_batch_size
        self.metatrain_label_ratio = FLAGS.metatrain_label_ratio
        self.metatest_label_ratio = FLAGS.metatest_label_ratio
        self.meta_num_pos_nodes = FLAGS.meta_num_pos_nodes
        self.meta_num_neg_nodes = FLAGS.meta_num_neg_nodes

        self._load_basic_data()
        self._split_data()
        if training:
            self._load_context_pairs()
            self._load_sampling_table()
            self._build_tf_dataset()

    def get_unsup_data(self):
        if not hasattr(self, "unsup_iterator"):
            self.unsup_iterator = iter(self.tf_dataset_unsupervised)
        batch_data = next(self.unsup_iterator)
        batch_data = {"unsup": batch_data}
        batch_data["unsup"]["nodes_neg"] = tf.convert_to_tensor(
            self.negative_sampling(), dtype=tf.int32)
        return batch_data

    def get_data(self, mode="train", num_tasks=None, test_num_pos=None, test_num_neg=None, test_num=None):
        if mode == "train":
            if not hasattr(self, "iterator"):
                self.iterator = iter(self.tf_dataset)
            batch_data = next(self.iterator)
            batch_data["unsup"]["nodes_neg"] = tf.convert_to_tensor(
                self.negative_sampling(), dtype=tf.int32)
        else:
            assert num_tasks is not None
            assert mode == "valid" or mode == "test"
            batch_data = {}
            batch_data["meta"] = self.get_tasks(mode, num_tasks, test_num_pos=test_num_pos, test_num_neg=test_num_neg)
        return batch_data

    def get_tasks(self, mode, num_tasks=None, rng=None, test_num_pos=None, test_num_neg=None):
        rng = rng or self.rng_tasks
        if mode == "train":
            num_tasks = self.meta_batch_size
            sample_from = self.metatrain_labels
        else:
            assert num_tasks is not None
            num_tasks = num_tasks
            if mode == "valid":
                if hasattr(self, "_metaval_tasks") and \
                        len(self._metaval_tasks["label_id"]) == num_tasks:
                    return self._metaval_tasks
                sample_from = self.metaval_labels
            elif mode == "test":
                if hasattr(self, "_metatest_tasks") and \
                        len(self._metatest_tasks["label_id"]) == num_tasks:
                    return self._metatest_tasks
                sample_from = self.metatest_labels

        tasks = sample_tasks(
            rng, num_tasks,
            self.meta_num_pos_nodes, self.meta_num_neg_nodes,
            sample_from, self.G)
        tasks = batching_tasks(tasks)
        tasks = nest.map_structure(lambda x: tf.convert_to_tensor(x), tasks)
        tasks = dict(tasks._asdict())
        if mode == "valid":
            self._metaval_tasks = tasks
        elif mode == "test":
            self._metatest_tasks = tasks

        return tasks

    def _build_tf_dataset(self):
        self._build_tf_dataset_unsupervised()
        self._build_tf_dataset_meta()
        tf_dataset = tf.data.Dataset.zip({
            "unsup": self.tf_dataset_unsupervised,
            "meta": self.tf_dataset_meta})
        self.tf_dataset = tf_dataset.prefetch(1)

    def _build_tf_dataset_unsupervised(self):
        dataset_tensors = {
            "nodes_u": self.nodes_u,
            "nodes_v": self.nodes_v}
        tf_dataset = tf.data.Dataset.from_tensor_slices(dataset_tensors)
        if self.dataset == "Flickr":
            tf_dataset = tf_dataset.shuffle(750000)
        else:
            tf_dataset = tf_dataset.shuffle(self.nodes_u.shape[0])
        # tf_dataset = tf.data.Dataset.range(self.num_nodes)
        # tf_dataset = tf_dataset.map(lambda x: {"nodes": x})
        # tf_dataset = tf_dataset.shuffle(self.num_nodes)
        tf_dataset = tf_dataset.repeat()
        tf_dataset = tf_dataset.batch(self.unsup_batch_size)
        self.tf_dataset_unsupervised = tf_dataset

    def _build_tf_dataset_meta(self):
        def gen():
            rng = np.random.RandomState(self.seed)
            while True:
                yield self.get_tasks("train", rng=rng)
        tf_dataset = tf.data.Dataset.from_generator(
            gen, output_types={
                "label_id": tf.int64,
                "train_nodes": tf.int64, "train_labels": tf.int64,
                "test_nodes": tf.int64, "test_labels": tf.int64})
        self.tf_dataset_meta = tf_dataset

    def _load_basic_data(self):
        # load graph and obtain relevant information
        self.G = nx.read_gpickle(self.dataset_path)
        self.labels = self.G.graph["label_array"].astype(np.float32)
        self.num_nodes = self.G.number_of_nodes()
        self.adj = nx.convert_matrix.to_scipy_sparse_matrix(
            self.G,
            nodelist=list(range(self.G.number_of_nodes())),
            format="csr")
        self.degrees = np.asarray(self.adj.sum(axis=1),
                                  dtype=np.int64).flatten()

    def _load_context_pairs(self):
        self.nodes_u, self.nodes_v = self.adj.nonzero()

    def _load_sampling_table(self):
        self.sampling_table_file = osp.join(
            osp.dirname(osp.abspath(self.dataset_path)),
            "sampling_table.npy")
        if osp.isfile(self.sampling_table_file):
            self.sampling_table = np.load(self.sampling_table_file)
        else:
            self._generate_sampling_table()

    def _split_data(self):
        num_labels = self.labels.shape[1]
        num_metatrain_labels = int(num_labels * self.metatrain_label_ratio)
        num_metatest_labels = int(num_labels * self.metatest_label_ratio)

        shuffled_labels = self.rng_basic.permutation(num_labels)
        self.metatrain_labels, self.metatest_labels, self.metaval_labels = \
            np.split(shuffled_labels,
                     [num_metatrain_labels,
                      num_metatrain_labels + num_metatest_labels])

    def _generate_sampling_table(self):
        import math
        table_size = 1e8
        power = 0.75
        numNodes = self.num_nodes

        print("Pre-procesing for non-uniform negative sampling!")
        node_degree = self.degrees.copy()
        # node_degree = np.zeros(numNodes)  # out degree

        # import ipdb; ipdb.set_trace()
        # look_up = self.g.look_up_dict
        # for edge in self.g.G.edges():
        #     node_degree[look_up[edge[0]]
        #                 ] += self.g.G[edge[0]][edge[1]]["weight"]

        norm = sum([math.pow(node_degree[i], power) for i in range(numNodes)])

        self.sampling_table = np.zeros(int(table_size), dtype=np.uint32)

        p = 0
        i = 0
        for j in range(numNodes):
            p += float(math.pow(node_degree[j], power)) / norm
            while i < table_size and float(i) / table_size < p:
                self.sampling_table[i] = j
                i += 1
        np.save(self.sampling_table_file, self.sampling_table)

    def negative_sampling(self):
        negative_samples = []
        for _ in range(self.unsup_batch_size):
            random_indices = self.rng_basic.randint(
                0, self.sampling_table.shape[0],
                size=self.unsup_num_neg_samples)
            negative_samples.append(self.sampling_table[random_indices])
        negative_samples = np.asarray(negative_samples, dtype=np.int64)
        return negative_samples
