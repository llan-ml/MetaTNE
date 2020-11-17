# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

import ray
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from ray.tune.trainable import Trainable

from data_generator import DataGenerator
from my_model import MyModel
from loss_object import LossObject
from utils import flatten_dict, calculate_metrics, DotDict


class Trainer(Trainable):
    def _setup_tf_resource(self, gpu_id):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if ray.is_initialized():
            gpu_id = 0  # ray automatically set CUDA_VISIBLE_DEVICES for remote process
        else:
            gpu_id = gpu_id  # local run
        tf.config.experimental.set_visible_devices(gpus[gpu_id], "GPU")
        tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)

    def _setup(self, config):
        self.FLAGS = FLAGS = DotDict(config)
        self._setup_tf_resource(FLAGS.gpu)

        tf.random.set_seed(FLAGS.seed)
        self.rng = np.random.RandomState(FLAGS.seed)

        self.data_generator = DataGenerator(FLAGS=FLAGS)

        self.model = model = MyModel(FLAGS=FLAGS, num_nodes=self.data_generator.num_nodes)

        self.loss_object = LossObject(model=model, FLAGS=FLAGS)

        learning_rate_struc = FLAGS.learning_rate_struc
        learning_rate_meta = FLAGS.learning_rate_meta
        self.optimizer_struc = tf.keras.optimizers.Adam(learning_rate=learning_rate_struc)
        self.optimizer_meta = tf.keras.optimizers.Adam(learning_rate=learning_rate_meta)

        if FLAGS.autograph:
            self.train_one_step_struc = tf.function(self.train_one_step_struc)
            self.train_one_step_meta = tf.function(self.train_one_step_meta)
            self.ag_model = tf.function(self.model)
        else:
            self.ag_model = self.model

    def _train(self):
        FLAGS = self.FLAGS
        for _ in range(FLAGS.valid_step_period):
            threshold = (1.0 - 1.0 / (1.0 + self.FLAGS.decay_rate * np.floor(self.step / self.FLAGS.decay_steps)))
            loss_and_info = {
                "loss": {},
                "info": {"threshold": threshold},
                "metrics": {},
                "csv_fieldnames": self.csv_fieldnames
            }
            outputs = {}
            if self.rng.rand() > threshold:
                inputs = self.data_generator.get_unsup_data()
                struc_loss_and_info, struc_outputs = self.train_one_step_struc(inputs)
                outputs.update(struc_outputs)
                loss_and_info["loss"]["struc"] = struc_loss_and_info["loss"]
                loss_and_info["info"]["struc"] = struc_loss_and_info["info"]
                loss_and_info["info"]["struc_lr"] = (
                    self.optimizer_struc.lr(self.optimizer_struc.iterations - 1).numpy()
                    if callable(self.optimizer_struc.lr) else self.optimizer_struc.lr.numpy())
            else:
                inputs = self.data_generator.get_data()
                meta_loss_and_info, meta_outputs = self.train_one_step_meta(inputs)
                outputs.update(meta_outputs)
                loss_and_info["loss"]["meta"] = meta_loss_and_info["loss"]
                loss_and_info["info"]["meta"] = meta_loss_and_info["info"]
                loss_and_info["info"]["meta_lr"] = (
                    self.optimizer_meta.lr(self.optimizer_meta.iterations - 1).numpy()
                    if callable(self.optimizer_meta.lr) else self.optimizer_meta.lr.numpy())

            self._iteration += 1
            self._iterations_since_restore += 1

        self._iteration -= 1
        self._iterations_since_restore -= 1

        if "meta" in outputs:
            metrics = calculate_metrics(inputs["meta"]["test_labels"].numpy(),
                                        outputs["meta"]["logits"].numpy())
            loss_and_info["metrics"]["metatrain"] = metrics
        if (self.step + 1) % FLAGS.valid_step_period == 0:
            valid_metrics = self._test("valid")
            test_metrics = self._test("test")
            loss_and_info["metrics"]["metaval"] = valid_metrics
            loss_and_info["metrics"]["metatest"] = test_metrics

        def _func(x):
            try:
                y = float(x)
            except (TypeError, ValueError):
                y = x
            return y
        result = nest.map_structure(_func, loss_and_info)
        ahb_metric = (
            loss_and_info["metrics"]["metaval"]["roc_auc"] +
            loss_and_info["metrics"]["metaval"]["f1"])

        if not hasattr(self, "_best_ahb_metric"):
            self._best_ahb_metric = ahb_metric
            self._metric_after_best = 0.0
            self._step_after_best = 0
        if ahb_metric > self._best_ahb_metric:
            self._best_ahb_metric = ahb_metric
            self._metric_after_best = 0.0
            self._step_after_best = 0
            deviation_after_best = 0.0
        else:
            self._metric_after_best += ahb_metric
            self._step_after_best += 1
            deviation_after_best = self._best_ahb_metric - self._metric_after_best / self._step_after_best

        result["best_ahb_metric"] = self._best_ahb_metric
        result["deviation_after_best"] = deviation_after_best
        result["hpo_metric"] = self._best_ahb_metric - deviation_after_best
        return result

    def _test(self, mode, test_num_pos=None, test_num_neg=None):
        FLAGS = self.FLAGS
        num_tasks = 1000
        inputs = self.data_generator.get_data(mode, num_tasks)

        flat_inputs = nest.flatten(inputs)
        all_this_flat_outputs = []
        for i in range(0, num_tasks, FLAGS.meta_batch_size):
            # print(i, i+FLAGS.meta_batch_size)
            this_flat_inputs = [x[i: i + FLAGS.meta_batch_size] for x in flat_inputs]
            this_inputs = nest.pack_sequence_as(inputs, this_flat_inputs)
            this_outputs = self.ag_model(this_inputs, training=False, mode="meta")
            this_flat_outputs = nest.flatten(this_outputs)
            all_this_flat_outputs.append(this_flat_outputs)
        all_flat_outputs = list(zip(*all_this_flat_outputs))
        flat_outputs = [tf.concat(x, axis=0) for x in all_flat_outputs]
        outputs = nest.pack_sequence_as(this_outputs, flat_outputs)

        metrics = calculate_metrics(
            inputs["meta"]["test_labels"].numpy(),
            outputs["meta"]["logits"].numpy())
        return metrics

    def train_one_step_struc(self, inputs):
        with tf.name_scope("struc"):
            with tf.GradientTape() as tape:
                outputs = self.model(inputs, training=True, mode="struc")
                loss, info = self.loss_object.calculate_loss(outputs, inputs, mode="struc")

            with tf.name_scope("compute_grads"):
                weights_for_grad = [w for w in self.model.trainable_weights if "embedding" in w.name]
                gradients = tape.gradient(loss, weights_for_grad)
                grads_and_vars = [(grad, var) for grad, var
                                  in zip(gradients, weights_for_grad)
                                  if grad is not None]

            with tf.name_scope("apply_grads"):
                self.optimizer_struc.apply_gradients(grads_and_vars)

        ret = {"loss": loss, "info": info}, outputs
        return ret

    def train_one_step_meta(self, inputs):
        with tf.name_scope("meta"):
            with tf.GradientTape() as tape:
                outputs = self.model(inputs, training=True, mode="meta")
                loss, info = self.loss_object.calculate_loss(outputs, inputs, mode="meta")

            with tf.name_scope("compute_grads"):
                weights_for_grad = [w for w in self.model.trainable_weights]
                gradients = tape.gradient(loss, weights_for_grad)
                grads_and_vars = [(grad, var) for grad, var
                                  in zip(gradients, weights_for_grad)
                                  if grad is not None]

            with tf.name_scope("apply_grads"):
                self.optimizer_meta.apply_gradients(grads_and_vars)

        ret = {"loss": loss, "info": info}, outputs
        return ret

    @property
    def step(self):
        return self._iteration

    @property
    def csv_fieldnames(self):
        if hasattr(self, "_csv_fieldnames"):
            return self._csv_fieldnames
        result_structure = {
            "metrics": {
                "metatrain": {"roc_auc": None, "precision": None,
                              "recall": None, "f1": None},
                "metaval": {"roc_auc": None, "precision": None,
                            "recall": None, "f1": None},
                "metatest": {"roc_auc": None, "precision": None,
                             "recall": None, "f1": None}
            },
            "deviation_after_best": None,
            "training_iteration": None
        }
        fieldnames = list(flatten_dict(result_structure))
        self._csv_fieldnames = fieldnames
        return self._csv_fieldnames
