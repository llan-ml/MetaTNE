# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

from collections import namedtuple

import numpy as np
import scipy as sp
import networkx as nx
import tensorflow as tf
from tensorflow.python.util import nest
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics


TaskInfo = namedtuple("TaskInfo", ["label_id",
                                   "train_nodes", "train_labels",
                                   "test_nodes", "test_labels"])


def batching_tasks(tasks):
    assert isinstance(tasks, list)
    assert isinstance(tasks[0], TaskInfo)
    new = {}
    for attr in ["label_id", "train_nodes", "train_labels",
                 "test_nodes", "test_labels"]:
        new[attr] = np.stack([getattr(t, attr) for t in tasks])
    batch_tasks = TaskInfo(**new)
    return batch_tasks


def sample_tasks(
        rng,
        num_tasks,
        num_positive_nodes,
        num_negative_nodes,
        sample_from_labels,
        G,
        test_num_pos=None,
        test_num_neg=None):
    tasks = []
    while len(tasks) < num_tasks:
        tmp_task = sample_a_task(
            rng, num_positive_nodes, num_negative_nodes,
            sample_from_labels, G,
            test_num_pos, test_num_neg)
        if not any([compare_tasks(t, tmp_task) for t in tasks]):
            tasks.append(tmp_task)
        else:
            import ipdb; ipdb.set_trace()
    return tasks


def sample_a_task(
        rng,
        num_positive_nodes,
        num_negative_nodes,
        sample_from_labels,
        G,
        test_num_pos,
        test_num_neg):
    if test_num_pos is None:
        test_num_pos = num_positive_nodes
        test_num_neg = num_negative_nodes
    label_array = G.graph["label_array"]
    sampled_label = None
    while not sampled_label:
        tmp = rng.choice(sample_from_labels)
        if np.count_nonzero(label_array[:, tmp]) > num_positive_nodes + test_num_pos:
            sampled_label = tmp
    all_pos_nodes = np.nonzero(label_array[:, sampled_label] != 0)[0]
    all_neg_nodes = np.nonzero(label_array[:, sampled_label] == 0)[0]
    sampled_pos_nodes = rng.choice(all_pos_nodes,
                                   size=num_positive_nodes + test_num_pos,
                                   replace=False)
    sampled_neg_nodes = rng.choice(all_neg_nodes,
                                   size=num_negative_nodes + test_num_neg,
                                   replace=False)
    train_pos_nodes, test_pos_nodes = np.split(
        sampled_pos_nodes, [num_positive_nodes])
    train_neg_nodes, test_neg_nodes = np.split(
        sampled_neg_nodes, [num_negative_nodes])

    train_nodes = rng.permutation(
        np.concatenate((train_pos_nodes, train_neg_nodes)))
    train_labels = label_array[train_nodes, sampled_label]
    test_nodes = rng.permutation(
        np.concatenate((test_pos_nodes, test_neg_nodes)))
    test_labels = label_array[test_nodes, sampled_label]
    task = TaskInfo(label_id=sampled_label,
                    train_nodes=train_nodes,
                    train_labels=train_labels,
                    test_nodes=test_nodes,
                    test_labels=test_labels)
    return task

def sample_fixed_tasks(
        rng,
        num_tasks,
        num_positive_nodes,
        num_negative_nodes,
        sample_from_labels,
        G,
        test_num):
    tasks = []
    while len(tasks) < num_tasks:
        label_array = G.graph["label_array"]
        sampled_label = None
        while not sampled_label:
            tmp = rng.choice(sample_from_labels)
            if np.count_nonzero(label_array[:, tmp]) > num_positive_nodes:
                sampled_label = tmp
        all_pos_nodes = np.nonzero(label_array[:, sampled_label] != 0)[0]
        all_neg_nodes = np.nonzero(label_array[:, sampled_label] == 0)[0]
        sampled_pos_nodes = rng.choice(all_pos_nodes,
                                       size=num_positive_nodes,
                                       replace=False)
        sampled_neg_nodes = rng.choice(all_neg_nodes,
                                       size=num_negative_nodes,
                                       replace=False)
        # train_pos_nodes, test_pos_nodes = np.split(
        #     sampled_pos_nodes, [num_positive_nodes])
        # train_neg_nodes, test_neg_nodes = np.split(
        #     sampled_neg_nodes, [num_negative_nodes])

        train_nodes = rng.permutation(
            np.concatenate((sampled_pos_nodes, sampled_neg_nodes)))
        train_labels = label_array[train_nodes, sampled_label]
        # test_nodes = rng.permutation(
        #     np.concatenate((test_pos_nodes, test_neg_nodes)))

        done = False
        while not done:
            test_nodes = []
            while len(test_nodes) < test_num:
                tmp = rng.randint(label_array.shape[0])
                if tmp not in train_nodes:
                    test_nodes.append(tmp)
            test_nodes = np.asarray(test_nodes)
            test_labels = label_array[test_nodes, sampled_label]
            if np.sum(test_labels) > 0:
                done = True
        task = TaskInfo(label_id=sampled_label,
                        train_nodes=train_nodes,
                        train_labels=train_labels,
                        test_nodes=test_nodes,
                        test_labels=test_labels)

        if not any([compare_tasks(t, task) for t in tasks]):
            tasks.append(task)
        else:
            import ipdb; ipdb.set_trace()
    return tasks


def sample_random_tasks(
        rng,
        num_tasks,
        num_positive_nodes,
        num_negative_nodes,
        G):
    tasks = []
    while len(tasks) < num_tasks:
        random_nodes = rng.choice(G.number_of_nodes(),
                                  size=2 * (num_positive_nodes + num_negative_nodes),
                                  replace=False)
        train_nodes, test_nodes = np.split(random_nodes, 2)
        train_labels, test_labels = np.zeros_like(train_nodes), np.zeros_like(test_nodes)
        train_labels[rng.permutation(train_labels.shape[0])[:num_positive_nodes]] = 1
        test_labels[rng.permutation(test_labels.shape[0])[:num_positive_nodes]] = 1
        tmp_task = TaskInfo(label_id=-1,
                            train_nodes=train_nodes, train_labels=train_labels,
                            test_nodes=test_nodes, test_labels=test_labels)
        if not any([compare_tasks(t, tmp_task) for t in tasks]):
            tasks.append(tmp_task)
        else:
            import ipdb; ipdb.set_trace()
    return tasks


def compare_tasks(task_1, task_2):
    if task_1.label_id != task_2.label_id:
        return False
    else:
        for attr in ["train_nodes", "test_nodes"]:
            if set(getattr(task_1, attr)) == set(getattr(task_2, attr)):
                continue
            else:
                return False
        return True


def _sample_tasks(rng,
                  num_tasks,
                  num_labels_per_task,
                  num_shots_per_label,
                  sample_from_labels,
                  label_array):
    all_tasks = []
    while len(all_tasks) < num_tasks:
        sampled_labels = rng.choice(sample_from_labels,
                                    size=num_labels_per_task,
                                    replace=False)
        associated_nodes = set()
        for label in sampled_labels:
            nodes = label_array[:, label].nonzero()[0]
            associated_nodes.update(nodes)
        associated_nodes = np.asarray(list(associated_nodes), dtype=np.int64)

        num_sampled_nodes = num_labels_per_task * num_shots_per_label
        rng.shuffle(associated_nodes)
        train_nodes = associated_nodes[:num_sampled_nodes]
        test_nodes = associated_nodes[num_sampled_nodes: num_sampled_nodes * 2]
        if np.all(label_array[train_nodes][:, sampled_labels].sum(0)) \
                and np.all(label_array[test_nodes][:, sampled_labels].sum(0)):
            task = TaskInfo(labels=sampled_labels,
                            train_nodes=train_nodes,
                            test_nodes=test_nodes)
            if not any([compare_tasks(x, task) for x in all_tasks]):
                all_tasks.append(task)
            else:
                import ipdb; ipdb.set_trace()
            print(f"{len(all_tasks)}")

    return all_tasks


def calculate_metrics(labels, logits):
    true_labels = labels
    pred_probs = tf.nn.softmax(logits, axis=-1).numpy()
    pred_scores = pred_probs[:, :, 1]
    pred_labels = np.argmax(logits, axis=-1)
    # pred_scores = tf.sigmoid(logits).numpy().squeeze()
    # pred_labels = np.asarray(pred_scores > 0.5, np.int64)
    accs = []
    # baccs = []
    aucs = []
    precisions = []
    recalls = []
    f1s = []
    # for y_true, y_pred in zip(true_labels, pred_labels):
    for y_true, y_pred, y_score in zip(true_labels, pred_labels, pred_scores):
        accs.append(metrics.accuracy_score(y_true, y_pred))
        # baccs.append(metrics.balanced_accuracy_score(y_true, y_pred))
        aucs.append(metrics.roc_auc_score(y_true, y_score))
        precisions.append(metrics.precision_score(y_true, y_pred))
        recalls.append(metrics.recall_score(y_true, y_pred))
        f1s.append(metrics.f1_score(y_true, y_pred))
    acc = np.mean(accs)
    auc = np.mean(aucs)
    precison = np.mean(precisions)
    recall = np.mean(recalls)
    f1 = np.mean(f1s)
    outputs = {
        # "accuracy": acc,
        "roc_auc": auc,
        "precision": precison,
        "recall": recall,
        "f1": f1
    }
    return outputs


def flatten_dict(dt, separator="/"):
    dt = dt.copy()
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[separator.join([key, subkey])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, item):
        value = dict.__getattribute__(self, item)
        if isinstance(value, dict):
            return DotDict(value)
        else:
            return value


if __name__ == "__main__":
    data = "BlogCatalog"
    rng = np.random.RandomState(123)
    G = nx.read_gpickle(f"./dataset/{data}/{data}_nx.pkl")
    # label_array = G.graph["label_array"]
    # num_labels = label_array.shape[1]
    # rng = np.random.RandomState(123)
    # shuffled_labels = rng.permutation(num_labels)
    # metatrain_labels = shuffled_labels[:23]
    # metaval_labels = shuffled_labels[23: -8]
    # metatest_labels = shuffled_labels[-8:]
    # metatrain_tasks = sample_tasks(rng, 10, 5, 5, metatrain_labels, label_array)
    # nodes = list(G.nodes())
    # pairs = run_random_walks(G, nodes, 50, 5)
    # with open(f"./dataset/{data}/{data}_context_pairs.txt", "w") as f:
    #     f.write("\n".join([f"{p[0]},{p[1]}" for p in pairs]))
    tasks = sample_tasks(
        rng, 100, 10, 10,
        rng.choice(G.graph["label_array"].shape[1], size=20, replace=False),
        G)
