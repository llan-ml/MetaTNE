# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

import tensorflow as tf
from tensorflow.python.keras import layers as keras_layers


class Attention(keras_layers.Layer):
    def __init__(self, hidden_size, output_size, num_heads, attention_dropout):
        if hidden_size % num_heads:
            raise ValueError(
                f"Hidden size ({hidden_size}) must be divisible "
                f"by the number of heads ({num_heads}).")

        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout

        self.q_dense_layer = keras_layers.Dense(
            hidden_size, use_bias=False, name="q")
        self.k_dense_layer = keras_layers.Dense(
            hidden_size, use_bias=False, name="k")
        self.v_dense_layer = keras_layers.Dense(
            hidden_size, use_bias=False, name="v")
        self.output_dense_layer = keras_layers.Dense(
            output_size, use_bias=False, name="output_transform")

    def split_heads(self, x):
        with tf.name_scope("split_heads"):
            input_shape = tf.shape(x)
            batch_size, length, length_tmp = input_shape[0], input_shape[1], input_shape[2]
            depth = (self.hidden_size // self.num_heads)
            x = tf.reshape(x, [batch_size, length, length_tmp, self.num_heads, depth])
            return tf.transpose(x, [0, 1, 3, 2, 4])

    def combine_heads(self, x):
        with tf.name_scope("combine_heads"):
            input_shape = tf.shape(x)
            batch_size, length, legnth_tmp = input_shape[0], input_shape[1], input_shape[3]
            x = tf.transpose(x, [0, 1, 3, 2, 4])
            return tf.reshape(x, [batch_size, length, legnth_tmp, self.hidden_size])

    def call(self, x, y, training):
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        logits = tf.matmul(q, k, transpose_b=True)
        weights = tf.nn.softmax(logits, name="attention_weights")
        if training:
            weights = tf.nn.dropout(weights, rate=self.attention_dropout)
        attention_output = tf.matmul(weights, v)
        attention_output = self.combine_heads(attention_output)
        attention_output = self.output_dense_layer(attention_output)
        return attention_output


class SelfAttention(Attention):
    def call(self, x, training):
        return Attention.call(self, x, x, training=training)
