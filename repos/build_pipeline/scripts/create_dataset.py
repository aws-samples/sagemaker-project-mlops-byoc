import argparse
import pathlib
import json
import pandas as pd
import numpy as np
import os

# Parse argument variables passed via the CreateDataset processing step
parser = argparse.ArgumentParser()
parser.add_argument("--athena-data", type=str)
args = parser.parse_args()

dataset = pd.read_parquet(args.athena_data, engine="pyarrow")
train = dataset[dataset["data_type"]=="train"]
test = dataset.drop(train.index)

# ####################################################################
# Provide your own logics to generate train data!!!!!!!
# ####################################################################

"""
train:
    pos:
        xxx.txt
    neg:
        xxx.txt
val:
    pos:
        xxx.txt
    net:
        xxx.txt
"""

def gen_train_val(df, save_dir):
    for index, row in df.iterrows():
        file_name = row["index"]
        label = row["label"]
        file_dir = save_dir / label
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        
        with open(f"{file_dir}/{file_name}.txt", 'w') as f:
            f.write(row["text"])

# Write train, test splits to output path
train_output_path = pathlib.Path("/opt/ml/processing/output/train")
test_output_path = pathlib.Path("/opt/ml/processing/output/test")
#baseline_path = pathlib.Path("/opt/ml/processing/output/baseline")

gen_train_val(train, train_output_path)
gen_train_val(test, test_output_path)

# Save baseline with headers
# train.to_csv(baseline_path / "baseline.csv", index=False, header=True)