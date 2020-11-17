# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

import os
import csv
import json
import distutils.version
import numpy as np
import tensorflow as tf
import ray.cloudpickle as cloudpickle
from ray.tune.result import NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S, \
    TIMESTEPS_TOTAL, DEFAULT_RESULTS_DIR
from ray.tune import logger as tune_logger
from ray.tune.utils import flatten_dict
from ray.tune.trial import Trial


def to_tf_values(result, path):
    values = []
    type_list = [int, float, np.float32, np.float64, np.int32, np.int64]
    for attr, value in result.items():
        if value is not None:
            if type(value) in type_list:
                values.append((
                    "/".join(path + [attr]), value))
            elif type(value) is dict:
                values.extend(to_tf_values(value, path + [attr]))
    return values


def tf2_logger_creator_factory(identifier, local_dir=None):
    local_dir = local_dir or DEFAULT_RESULTS_DIR
    def tf2_logger_creator(config):
        logdir = Trial.create_logdir(identifier, local_dir)
        logger = tune_logger.UnifiedLogger(config, logdir, loggers=LOGGERS)
        # logger = TF2Logger(config, logdir)
        return logger
    return tf2_logger_creator


class TF2Logger(tune_logger.Logger):
    def _init(self):
        use_tf2_api = distutils.version.LooseVersion(tf.version.VERSION) >= \
            distutils.version.LooseVersion("2.0")
        if not use_tf2_api:
            raise ImportError
        self.update_config(self.config)
        self._file_writer = None

    def on_result(self, result):
        if self._file_writer is None:
            self._file_writer = tf.summary.create_file_writer(self.logdir)
        tmp = result.copy()
        for k in [
            "config", "pid", "timestamp", TIME_TOTAL_S, TRAINING_ITERATION
        ]:
            if k in tmp:
                del tmp[k]
        for k in list(tmp.keys()):
            if "since" in k:
                del tmp[k]
        # values = to_tf_values(tmp, ["ray", "tune"])
        values = to_tf_values(tmp, [])
        t = result.get(TIMESTEPS_TOTAL) or result[TRAINING_ITERATION]
        with tf.device("/device:CPU:0"):
            with self._file_writer.as_default():
                for attr, value in values:
                    tf.summary.scalar(attr, value, step=t)
                self._file_writer.flush()

    def flush(self):
        if self._file_writer is not None:
            self._file_writer.flush()

    def close(self):
        if self._file_writer is not None:
            self._file_writer.close()

    def update_config(self, config):
        self.config = config
        config_out = os.path.join(self.logdir, "params.json")
        with open(config_out, "w") as f:
            json.dump(
                self.config,
                f,
                indent=2,
                sort_keys=True,
                cls=tune_logger._SafeFallbackEncoder)
        config_pkl = os.path.join(self.logdir, "params.pkl")
        with open(config_pkl, "wb") as f:
            cloudpickle.dump(self.config, f)


class CSVLogger(tune_logger.CSVLogger):
    def on_result(self, result):
        tmp = result.copy()
        if "config" in tmp:
            del tmp["config"]
        if "csv_fieldnames" in tmp:
            csv_fieldnames = tmp.pop("csv_fieldnames")
        else:
            csv_fieldnames = None
        result = flatten_dict(tmp, delimiter="/")
        if self._csv_out is None:
            self._csv_out = csv.DictWriter(self._file,
                                           csv_fieldnames or result.keys())
            if not self._continuing:
                self._csv_out.writeheader()
        self._csv_out.writerow(
            {k: v
             for k, v in result.items() if k in self._csv_out.fieldnames})
        self._file.flush()


LOGGERS = (CSVLogger, TF2Logger)
