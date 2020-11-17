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

from transformer.attention import SelfAttention
from transformer.ffn import FeedForwardNetwork


class Transformer(keras_layers.Layer):
    def __init__(self, hparams, name=None):
        super(Transformer, self).__init__(name=name)
        self.hparams = hparams

        self.encoder_stack = EncoderStack(hparams)

    def call(self, inputs, training):
        with tf.name_scope("Transformer"):
            # if training:
            #     inputs = tf.nn.dropout(
            #         inputs, rate=self.hparams["layer_postprocess_dropout"])
            return self.encoder_stack(inputs, training=training)


class LayerNormalization(keras_layers.Layer):
    """Applies layer normalization."""
    def __init__(self, hidden_size):
        super(LayerNormalization, self).__init__()
        self.hidden_size = hidden_size

    def build(self, input_shape):
        """Builds the layer."""
        # Passing experimental_autocast=False causes these variables to not be
        # automatically casted to fp16 when mixed precision is used. Since we use
        # float32 in call() for numeric stability, we do not want variables to be
        # casted to fp16.
        self.scale = self.add_weight(
            "layer_norm_scale",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.ones_initializer(),
            experimental_autocast=False)
        self.bias = self.add_weight(
            "layer_norm_bias",
            shape=[self.hidden_size],
            dtype="float32",
            initializer=tf.zeros_initializer(),
            experimental_autocast=False)
        super(LayerNormalization, self).build(input_shape)

    def get_config(self):
        return {
            "hidden_size": self.hidden_size,
        }

    def call(self, x, epsilon=1e-6):
        input_dtype = x.dtype
        if input_dtype == tf.float16:
            x = tf.cast(x, tf.float32)
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.math.rsqrt(variance + epsilon)
        return tf.cast(norm_x * self.scale + self.bias, input_dtype)


class PrePostProcessingWrapper(keras_layers.Wrapper):
    def __init__(self, layer, hparams):
        super(PrePostProcessingWrapper, self).__init__(layer=layer)
        self.hparams = hparams
        self.postprocess_dropout = hparams["layer_postprocess_dropout"]
        self.layer_norm = LayerNormalization(hparams["output_size"])

    # def call(self, inputs, training, *args, **kwargs):
    def call(self, inputs, training):
        # training = kwargs["training"]
        normed_inputs = self.layer_norm(inputs)
        output = self.layer(normed_inputs, training=training)
        if training:
            output = tf.nn.dropout(output, rate=self.postprocess_dropout)
        return inputs + output
        # return output


class EncoderStack(keras_layers.Layer):
    def __init__(self, hparams):
        super(EncoderStack, self).__init__()
        self.hparams = hparams
        self.with_ffn = hparams["with_ffn"]
        self.layers = []

        for _ in range(hparams["num_hidden_layers"]):
            attention_layer = SelfAttention(
                hparams["hidden_size"], hparams["output_size"],
                hparams["num_heads"],
                hparams["attention_dropout"])
            if self.with_ffn:
                ffn_layer = FeedForwardNetwork(
                    hparams["output_size"], hparams["filter_size"],
                    hparams["relu_dropout"])

                self.layers.append([
                    PrePostProcessingWrapper(attention_layer, hparams),
                    PrePostProcessingWrapper(ffn_layer, hparams)
                ])
            else:
                self.layers.append([
                    PrePostProcessingWrapper(attention_layer, hparams)
                ])

        self.output_normalization = LayerNormalization(hparams["output_size"])

    def call(self, encoder_inputs, training):
        output = encoder_inputs
        for n, layer in enumerate(self.layers):
            attention_layer = layer[0]
            if self.with_ffn:
                ffn_layer = layer[1]

            with tf.name_scope(f"layer_{n}"):
                with tf.name_scope("attention"):
                    output = attention_layer(
                        output, training=training)
                if self.with_ffn:
                    with tf.name_scope("ffn"):
                        output = ffn_layer(output, training=training)
        return self.output_normalization(output)
