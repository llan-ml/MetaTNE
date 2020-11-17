# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify
# it under the terms of the Apache License Version 2.0.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License Version 2.0 for more details.
# ============================================================================

import sys
import pandas as pd
from absl import flags
from ray.tune.analysis import Analysis

FLAGS = flags.FLAGS

flags.DEFINE_string("logdir", "", "")
FLAGS(sys.argv)


analysis = Analysis(experiment_dir=FLAGS.logdir)
all_configs = analysis.get_all_configs()
rows = {}
for path, df in analysis.trial_dataframes.items():
    df = df.filter(regex="val|test|iteration").assign(
        metric=lambda x: x.filter(regex="val/(?:roc_auc|f1)").sum(axis=1))
    idx = df["metric"].idxmax()
    rows[path] = df.iloc[idx].to_dict()
    rows[path]["seed"] = all_configs[path]["seed"]

df = pd.DataFrame.from_dict(rows, orient="index")
df = df.filter(regex="test/(?:roc_auc|f1|recall)").rename(
    mapper=lambda x: x.split("/")[-1], axis=1).rename(
    columns={"roc_auc": "AUC", "f1": "F1", "recall": "Recall"})

df_stats = df.describe().loc[["mean", "std"]]
print(df_stats[["AUC", "F1", "Recall"]])
