# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================


CONFIGS = {
    "BlogCatalog": {
        "unsup_batch_size": 2048,
        "meta_batch_size": 64,
        "weight_decay": 0.01,
        "learning_rate_struc": 0.001,
        "learning_rate_meta": 0.001,
        "decay_rate": 0.1,
        "decay_steps": 500,
        "hidden_size": 256,
        "num_heads": 4,
        "filter_size": 256,
        "num_hidden_layers": 1
    },
    "PPI": {
        "unsup_batch_size": 1024,
        "meta_batch_size": 128,
        "weight_decay": 0.1,
        "learning_rate_struc": 0.0001,
        "learning_rate_meta": 0.001,
        "decay_rate": 0.1,
        "decay_steps": 1000,
        "hidden_size": 256,
        "num_heads": 1,
        "filter_size": 256,
        "num_hidden_layers": 1
    },
    "Mashup": {
        "unsup_batch_size": 2048,
        "meta_batch_size": 32,
        "weight_decay": 0.1,
        "learning_rate_struc": 0.001,
        "learning_rate_meta": 0.0001,
        "decay_rate": 0.1,
        "decay_steps": 500,
        "hidden_size": 256,
        "num_heads": 1,
        "filter_size": 512,
        "num_hidden_layers": 1
    },
    "Flickr": {
        "unsup_batch_size": 2048,
        "meta_batch_size": 64,
        "weight_decay": 0.1,
        "learning_rate_struc": 0.001,
        "learning_rate_meta": 0.0001,
        "decay_rate": 0.1,
        "decay_steps": 2000,
        "hidden_size": 256,
        "num_heads": 4,
        "filter_size": 512,
        "num_hidden_layers": 1
    }
}
