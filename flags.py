# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

import sys
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("gpu", 0, "")
flags.DEFINE_integer("seed", 123, "")
flags.DEFINE_string("dataset_str", "BlogCatalog", "")

# Structure learning parameters
flags.DEFINE_integer("unsup_batch_size", 2048, "")
flags.DEFINE_integer("unsup_num_neg_samples", 5, "")

# Meta-learning parameters
flags.DEFINE_integer("meta_batch_size", 64, "")
flags.DEFINE_float("metatrain_label_ratio", 0.6, "")
flags.DEFINE_float("metatest_label_ratio", 0.2, "")
flags.DEFINE_integer("meta_num_pos_nodes", 10, "")
flags.DEFINE_integer("meta_num_neg_nodes", 20, "")

# Loss parameters
flags.DEFINE_float("weight_decay", 0.01, "")

# Training parameters
flags.DEFINE_float("learning_rate_struc", 0.001, "")
flags.DEFINE_float("learning_rate_meta", 0.001, "")
flags.DEFINE_integer("max_steps", 100, "The total steps (*valid_step_period)")
flags.DEFINE_float("decay_rate", 0.1, "")
flags.DEFINE_float("decay_steps", 500, "")
flags.DEFINE_float("meta_prob", 0.5, "")
flags.DEFINE_boolean("autograph", True, "")
flags.DEFINE_integer("print_step_period", 10, "")
flags.DEFINE_integer("valid_step_period", 200, "")

# Model parameters
flags.DEFINE_integer("hidden_size", 256, "")
flags.DEFINE_integer("output_size", 128, "")
flags.DEFINE_integer("num_heads", 4, "")
flags.DEFINE_integer("filter_size", 256, "")
flags.DEFINE_integer("num_hidden_layers", 1, "")
flags.DEFINE_boolean("with_ffn", True, "")
flags.DEFINE_float("dropout_rate", 0.1, "")

# FLAGS(sys.argv)


def flag_dict():
    FLAGS(sys.argv)
    _flags = FLAGS.get_key_flags_for_module(
        FLAGS.find_module_defining_flag("gpu"))
    _flag_dict = {}
    for flag in _flags:
        print(f"{flag.name}: {flag.value}")
        _flag_dict[flag.name] = flag.value
    sys.stdout.flush()
    return _flag_dict
