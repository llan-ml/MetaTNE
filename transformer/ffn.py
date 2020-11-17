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


class FeedForwardNetwork(keras_layers.Layer):
    def __init__(self, hidden_size, filter_size, relu_dropout):
        super(FeedForwardNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout

        self.filter_dense_layer = keras_layers.Dense(
            filter_size,
            use_bias=True,
            activation=tf.nn.relu,
            name="filter_layer")
        self.output_dense_layer = keras_layers.Dense(
            hidden_size, use_bias=True, name="output_layer")

    def call(self, x, training):
        """
        Args:
            x: A tensor with shape [batch_size, length, hidden_size]
            training (boolean): whether in training mode or not.

        Returns:
            Output of the feedforward network.
            tensor with shape [batch_size, length, hidden_size]
        """
        # input_shape = tf.shape(x)
        # batch_size, length = input_shape[0], input_shape[1]

        output = self.filter_dense_layer(x)
        if training:
            output = tf.nn.dropout(output, rate=self.relu_dropout)
        output = self.output_dense_layer(output)
        return output
