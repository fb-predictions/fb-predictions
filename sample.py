
import pandas as pd
import numpy as np

import argparse
parser = argparse.ArgumentParser(description='Process input')

parser.add_argument('-p', dest='path', type=str, default="processed_data/all_validation.csv", required=False)
parser.add_argument('-s', dest='saving_path', type=str, default="data_exp_out/sample.csv", required=False)
parser.add_argument('-t', dest='type', type=str, default="uniform", required=False)
parser.add_argument('-n', dest='size', type=int, default=15, required=False)

args = parser.parse_args()

df_val = pd.read_csv(args.path)


if args.type == "abs_error":
    df_val["abs_error"] = np.abs(df_val["eng_rate"] - df_val["predictions"])
    df_val["error"] = df_val["eng_rate"] - df_val["predictions"]
    sum = np.sum(df_val["abs_error"])
    p = df_val["abs_error"] / sum
    posts = np.random.choice(df_val.index.to_numpy(), size=args.size, p=p.to_numpy())
    samples = df_val.iloc[posts]
    samples.to_csv(f"{args.saving_path}")


else:
    sample = df_val.sample(n=args.size)
    sample.to_csv(f"{args.saving_path}")
    sample = sample.drop(["score", "total_reactions", "shares", "comments", "likes", "eng_rate", "total_eng"], 1)
    sample.to_csv(f"{args.saving_path}")

