# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

import tensorflow as tf
from tensorflow.python.keras import models as keras_models

from layers.embedding import Embedding
from transformer.transformer import Transformer


class MyModel(keras_models.Model):
    def __init__(self, FLAGS, num_nodes):
        super(MyModel, self).__init__()

        self.embedding_layer_2nd = Embedding(num_nodes, FLAGS.output_size)
        self.embedding_layer_2nd_context = Embedding(num_nodes, FLAGS.output_size)

        transformer_hparams = {
            "hidden_size": FLAGS.hidden_size,
            "output_size": FLAGS.output_size,
            "num_heads": FLAGS.num_heads,
            "filter_size": FLAGS.filter_size,
            "num_hidden_layers": FLAGS.num_hidden_layers,
            "with_ffn": FLAGS.with_ffn,
            "attention_dropout": FLAGS.dropout_rate,
            "relu_dropout": FLAGS.dropout_rate,
            "layer_postprocess_dropout": FLAGS.dropout_rate,

        }

        self.transformer = Transformer(transformer_hparams)

    def call(self, inputs, training, mode):
        if mode == "struc":
            return self._struc_call(inputs)
        elif mode == "meta":
            return self._meta_call(inputs, training=training)
        elif mode == "all":
            struc_outputs = self._struc_call(inputs)
            meta_outputs = self._meta_call(inputs, training=training)
            outputs = {**struc_outputs, **meta_outputs}
            return outputs
        else:
            raise ValueError

    def _struc_call(self, inputs):
        unsup_inputs = inputs["unsup"]
        nodes_u = unsup_inputs["nodes_u"]
        nodes_v = unsup_inputs["nodes_v"]
        nodes_neg = unsup_inputs["nodes_neg"]

        embeddings_u_2nd = self.embedding_layer_2nd(nodes_u)
        embeddings_v_2nd = self.embedding_layer_2nd_context(nodes_v)
        embeddings_neg_2nd = self.embedding_layer_2nd_context(nodes_neg)
        return {
            "unsup": {
                "2nd": {
                    "nodes_u": embeddings_u_2nd,
                    "nodes_v": embeddings_v_2nd,
                    "nodes_neg": embeddings_neg_2nd
                }
            }
        }

    def _meta_call(self, inputs, training):
        meta_inputs = inputs["meta"]
        train_nodes = meta_inputs["train_nodes"]
        train_labels = meta_inputs["train_labels"]
        neg_train_nodes, pos_train_nodes = tf.dynamic_partition(
            train_nodes, tf.cast(train_labels, tf.int32), 2)

        neg_train_nodes = tf.reshape(neg_train_nodes,
                                     (train_nodes.shape[0], -1))
        pos_train_nodes = tf.reshape(pos_train_nodes,
                                     (train_nodes.shape[0], -1))

        neg_train_embeddings = self.embedding_layer_2nd(neg_train_nodes)
        pos_train_embeddings = self.embedding_layer_2nd(pos_train_nodes)

        test_nodes = meta_inputs["test_nodes"]
        test_embeddings = self.embedding_layer_2nd(test_nodes)

        neg_train_embeddings_for_att = tf.tile(tf.expand_dims(neg_train_embeddings, axis=1), multiples=[1, tf.shape(test_embeddings)[1], 1, 1])
        pos_train_embeddings_for_att = tf.tile(tf.expand_dims(pos_train_embeddings, axis=1), multiples=[1, tf.shape(test_embeddings)[1], 1, 1])

        transformer_input_for_neg = tf.concat([neg_train_embeddings_for_att, tf.expand_dims(test_embeddings, axis=2)], axis=2)
        transformer_input_for_pos = tf.concat([pos_train_embeddings_for_att, tf.expand_dims(test_embeddings, axis=2)], axis=2)

        transformer_output_for_neg = self.transformer(transformer_input_for_neg, training=training)
        transformer_output_for_pos = self.transformer(transformer_input_for_pos, training=training)

        adapted_neg_train_embeddings, test_embeddings_for_neg = tf.split(
            transformer_output_for_neg, [tf.shape(neg_train_nodes)[1], 1], axis=2)
        adapted_pos_train_embeddings, test_embeddings_for_pos = tf.split(
            transformer_output_for_pos, [tf.shape(pos_train_nodes)[1], 1], axis=2)

        neg_proto = tf.reduce_mean(adapted_neg_train_embeddings, axis=2)
        pos_proto = tf.reduce_mean(adapted_pos_train_embeddings, axis=2)
        test_embeddings_for_neg = tf.squeeze(test_embeddings_for_neg, axis=2)
        test_embeddings_for_pos = tf.squeeze(test_embeddings_for_pos, axis=2)

        depth = self.transformer.hparams["output_size"]
        distance_neg = - tf.reduce_sum(tf.square(test_embeddings_for_neg - neg_proto), axis=2) / (depth ** 0.5)
        distance_pos = - tf.reduce_sum(tf.square(test_embeddings_for_pos - pos_proto), axis=2) / (depth ** 0.5)

        distance = tf.stack([distance_neg, distance_pos], axis=2)

        outputs = {
            "meta": {
                "logits": distance
            }
        }
        return outputs
