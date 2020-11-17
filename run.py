# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

import ray
import copy
import tensorflow as tf
from ray import tune
from ray.tune import grid_search
from ray.tune.utils import deep_update

import flags
from trainer import Trainer
from ray_logger import LOGGERS
from model_configs import CONFIGS


gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[0], "GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

FLAGS = flags.FLAGS
default_config = flags.flag_dict()
default_config["max_steps"] = 200

ray.init(num_gpus=4, object_store_memory=50000000000)

datasets = [FLAGS.dataset_str]
tune_kwargs = []
for data in datasets:
    this_config = copy.deepcopy(default_config)
    this_config = deep_update(this_config, CONFIGS[data], False, [])
    this_config["dataset_str"] = data
    this_config["seed"] = grid_search(list(range(1, 51)))
    max_steps = this_config["max_steps"] * this_config["valid_step_period"]
    num_pos, num_neg = FLAGS.meta_num_pos_nodes, FLAGS.meta_num_neg_nodes
    tune_kwargs.append({
        "run_or_experiment": Trainer,
        "name": f"{data}_{num_pos}_{num_neg}",
        "stop": {"training_iteration": max_steps},
        "config": this_config,
        "resources_per_trial": {"cpu": 0, "gpu": 0.2},
        "local_dir": "~/ray_results",
        "loggers": LOGGERS,
        "global_checkpoint_period": 600})

for kwargs in tune_kwargs:
    tune.run(**kwargs)
