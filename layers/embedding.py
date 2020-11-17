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


class Embedding(keras_layers.Layer):
    def __init__(self, vocab_size, hidden_size, dtype=None, init_value=None, name=None):
        super(Embedding, self).__init__(dtype=dtype, name=name)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.init_value = init_value
        if self.init_value is not None:
            value_shape = init_value.shape
            assert vocab_size == value_shape[0], hidden_size == value_shape[1]

    def build(self, input_shape):
        if self.init_value is None:
            init_width = 0.5 / self.hidden_size
            _initializer = tf.random_uniform_initializer(
                minval=-init_width, maxval=init_width)
        else:
            def _initializer(*args, **kwargs):
                return tf.constant(self.init_value, dtype=tf.float32)
        with tf.name_scope("embedding_and_softmax"):
            self.embeddings = self.add_weight(
                "weights",
                shape=[self.vocab_size, self.hidden_size],
                dtype=tf.float32,
                initializer=_initializer,
                trainable=True)
        keras_layers.Layer.build(self, input_shape)

    def call(self, inputs, mode="embedding"):
        if mode == "embedding":
            return self._embedding(inputs)
        elif mode == "linear":
            return self._linear(inputs)
        else:
            raise ValueError(f"mode {mode} is not valid.")

    def _embedding(self, inputs):
        # inputs: an int tensor with shape [batch_size, length]
        with tf.name_scope("embedding"):
            embeddings = tf.gather(self.embeddings, inputs)
            # embeddings = tf.nn.embedding_lookup(self.embeddings, inputs)
        return embeddings

    def _linear(self, inputs):
        # inputs: a float tensor with shape [batch_size, length, hidden_size]
        with tf.name_scope("presoftmax_linear"):
            input_shape = tf.shape(inputs)
            batch_size, length = input_shape[0], input_shape[1]

            x = tf.reshape(inputs, [-1, self.hidden_size])
            logits = tf.matmul(x, self.embeddings, transpose_b=True)

            return tf.reshape(logits, [batch_size, length, self.vocab_size])


if __name__ == "__main__":
    embedding_layer = EmbeddingLayer(100, 8)
