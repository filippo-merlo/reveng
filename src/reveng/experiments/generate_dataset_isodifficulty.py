"""
This code applies iso-difficulty transforms [ReflectEnv, RotateEnv, TransposeEnv, StartGoalSwap]
to the existing dataset baseline_grids

"""

import os
import pickle
import sys

sys.path.append(os.path.join("C:/users/hchen/dropbox/reveng", "src"))
from papers.papers_code.reveng.src.reveng.environment_generator.env_transformations import (
    ReflectEnv,
    RotateEnv,
    StartGoalSwap,
    TransposeEnv,
)

dataset_path = (
    "C:/users/hchen/dropbox/reveng/src/reveng/experiments/datasets/baseline_grids.pkl"
)
output_path = "C:/users/hchen/dropbox/reveng/src/reveng/experiments/datasets/isodifficulty_grids.pkl"

print("\n1. Loading datasets...")
print(f"   Dataset: {dataset_path}")
with open(dataset_path, "rb") as f:
    grids_dataset = pickle.load(f)
print(f"   Loaded {len(grids_dataset)} environments from dataset")

iso_dict = {}
for key in grids_dataset.keys():
    env = grids_dataset[key]
    iso_dict[key + "_RotateEnv"] = RotateEnv().apply(env)
    iso_dict[key + "_ReflectEnv"] = ReflectEnv().apply(env)
    iso_dict[key + "_TransposeEnv"] = TransposeEnv().apply(env)
    iso_dict[key + "_StartGoalSwap"] = StartGoalSwap().apply(env)

with open(output_path, "wb") as f:
    pickle.dump(iso_dict, f)
