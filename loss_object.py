# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

import re
import tensorflow as tf


class LossObject(object):
    def __init__(self, model, FLAGS):
        self.model = model
        self.weight_decay = FLAGS.weight_decay

    def calculate_loss(self, outputs, inputs, mode):
        if mode == "struc":
            with tf.name_scope("struc_loss"):
                unsup_loss, unsup_info = self._calculate_unsup_loss(
                    outputs["unsup"], inputs["unsup"])
                loss = unsup_loss
                info = unsup_info

        elif mode == "meta":
            with tf.name_scope("meta_loss"):
                meta_loss, meta_info = self._calculate_meta_loss(
                    outputs["meta"], inputs["meta"])
                reg_loss = self._calculate_reg_loss()
                loss = meta_loss + self.weight_decay * reg_loss
                info = {"meta": meta_info["meta"], "reg": reg_loss}

        else:
            raise ValueError
        return loss, info

    def _calculate_meta_loss(self, outputs, inputs):
        logits = outputs["logits"]
        labels = inputs["test_labels"]

        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        loss = tf.reduce_mean(losses)
        info = {"meta": {}}
        return loss, info

    def _calculate_reg_loss(self):
        self.reg_names = []
        reg_losses = []
        for var in self.model.trainable_variables:
            if re.search("kernel", var.name):
                self.reg_names.append(var.name[:var.name.find(":")])
                reg_losses.append(tf.nn.l2_loss(var))
        if reg_losses:
            reg_loss = tf.math.add_n(reg_losses)
        else:
            reg_loss = tf.constant(0.0, dtype=tf.float32)
        return reg_loss

    def _calculate_unsup_loss(self, outputs, inputs):

        def _calculate_loss(embed_u, embed_v, embed_neg):
            positive_losses = tf.math.log_sigmoid(
                tf.reduce_sum(tf.multiply(embed_u, embed_v), axis=1))
            positive_loss = - tf.reduce_mean(positive_losses)
            negative_losses = tf.reduce_mean(
                tf.math.log_sigmoid(
                    - tf.reduce_sum(
                        tf.multiply(tf.expand_dims(embed_u, 1),
                                    embed_neg),
                        axis=-1)),
                axis=-1)
            negative_loss = - tf.reduce_mean(negative_losses)
            return positive_loss + negative_loss

        loss_2nd = _calculate_loss(outputs["2nd"]["nodes_u"], outputs["2nd"]["nodes_v"], outputs["2nd"]["nodes_neg"])
        return loss_2nd, {"struc": {"2nd": loss_2nd}}
