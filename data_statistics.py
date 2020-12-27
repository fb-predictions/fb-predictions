import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from transformers import DistilBertTokenizer
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

import argparse
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('-p', dest='path', type=str, default="processed_data/all_train.csv", required=False)
parser.add_argument('-s', dest='saving_path', type=str, default="data_stats", required=False)
parser.add_argument('-st', dest='stat_type', type=str, default="hist", required=False)
parser.add_argument('-f', dest='feature', type=str, default="year", required=False)
parser.add_argument('-t', dest='target', type=str, default="posts", required=False)
parser.add_argument('-y', dest='by_year', type=str, default="no", required=False)
parser.add_argument('-b', dest='balance_data', type=str, default="no", required=False)

args = parser.parse_args()

path = args.path
saving_path = args.saving_path
stat_type = args.stat_type

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')


# relevant_features = ['tinder']
def plot_feature_histogram_by_year(df, feature):
    for year in range(2013, 2021):
        num_bins = 200
        n, bins, patches = plt.hist(df.loc[df['year'] == year][feature], num_bins, facecolor='blue', alpha=0.5)
        plt.xlabel(f'{feature}')
        plt.ylabel('#posts')

        plt.title(f"Histogram of {feature} of {year}")
        plt.savefig(f"{saving_path}/{feature}_hist_year_{year}.pdf")


def plot_feature_histogram(df, feature):
    num_bins = 200
    n, bins, patches = plt.hist(df[feature], num_bins, facecolor='blue', alpha=0.5,  range=[0, 0.125])
    plt.xlabel(f'{feature}')
    plt.ylabel('#posts')

    plt.title(f"Histogram of #posts in relation of {feature}")
    plt.savefig(f"{saving_path}/{feature}_hist.pdf")


def plot_correlations(df, feature, target):
    plt.scatter(df[feature], df[target], s=0.1, alpha=0.3)
    plt.xlabel(f'{feature}')
    plt.ylabel(f'{target}')
    plt.title(f"Scatter of {target} in correlation of {feature}")
    plt.savefig(f"{saving_path}/correlation_{feature}_{target}.pdf")
    plt.clf()
    plt.cla()


def plot_trendline(df, target, target_name, feature, feature_name):
    df = df.dropna(subset=[target, feature])
    x = df[feature]
    y = df[target]

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    plt.plot(x, p(x), "r--")
    plt.legend(["mean"])

    plt.scatter(x, y, s=5, alpha=1)

    plt.xlabel(f'{feature}')
    plt.ylabel(f'{target}')
    axes = plt.gca()
    axes.set_ylim([0, 0.25])
    plt.title(f"BERT {target_name} in correlation of {feature_name}")
    plt.savefig(f"{saving_path}/trendline_{feature}_{target}.pdf")




def find_unknown(df):
    print("Unknown Stats")

    df["UNK"] = 0
    for idx, row in df.iterrows():
        tokens = tokenizer.tokenize(row["post_text"])
        if "[UNK]" in tokens:
            df.loc[idx, "UNK"] = 1
        semi_unk = sum([1 for t in tokens if "##" in t])
        perc = semi_unk / row["words_count"]
        if perc >= 0.1:
            df.loc[idx, "UNK"] = 1
    print("no UNK", len(df[df["UNK"] == 0]))
    print("UNK", len(df[df["UNK"] == 1]))
    print("eng rate")
    print("no UNK", np.mean(df[df["UNK"] == 0]["eng_rate"]))
    print("UNK", np.mean(df[df["UNK"] == 1]["eng_rate"]))

    print("predictions")
    print("no UNK", np.mean(df[df["UNK"] == 0]["predictions"]))
    print("UNK", np.mean(df[df["UNK"] == 1]["predictions"]))

    print("abs_error")
    print("no UNK", np.mean(df[df["UNK"] == 0]["abs_error"]))
    print("UNK", np.mean(df[df["UNK"] == 1]["abs_error"]))




trigger_words = ["feel", "bad", "miss", "cry", "scream", "depress", "love", "crap", "slut", "whore", "fuck", "motherfuck", "sex", "cunt", "shit", "shag", "kill", "die", "rape", "beat", "hit", "bitch", "bollocks", "asshole", "bastard", "bugger", "bloody hell", "dick", "pussy", "piss", "twat", "suck", "disgust", "twat", "embarrassed"]
trigger_regex = re.compile(r"\b" + r"[^\s]*\b|\b".join(trigger_words) + r"\b|\bass\b", re.IGNORECASE)


def trigger(df):
    print("Trigger Words Stats")

    df["trigger"] = df['post_text'].str.contains(trigger_regex).astype("Int64")
    print("no trigger", len(df[df["trigger"] == 0]))
    print("trigger", len(df[df["trigger"] == 1]))

    print("eng rate")
    print("no trigger", np.mean(df[df["trigger"] == 0]["eng_rate"]))
    print("trigger", np.mean(df[df["trigger"] == 1]["eng_rate"]))

    print("predictions")
    print("no trigger", np.mean(df[df["trigger"] == 0]["predictions"]))
    print("trigger", np.mean(df[df["trigger"] == 1]["predictions"]))

    print("abs_error")
    print("no trigger", np.mean(df[df["trigger"] == 0]["abs_error"]))
    print("trigger", np.mean(df[df["trigger"] == 1]["abs_error"]))
    # plot_correlations(df, 'predictions')
    # plot_trendline(df, 'abs_error')

    # top 10%
    print("top 10% eng_rate")
    print("no trigger",
          np.mean(df[df["trigger"] == 0].nlargest(len(df[df["trigger"] == 0]) // 10, 'eng_rate')['eng_rate']))
    print("trigger",
          np.mean(df[df["trigger"] == 1].nlargest(len(df[df["trigger"] == 1]) // 10, 'eng_rate')['eng_rate']))


def groups(df):
    print(df["eng_rate"].describe())
    bins = [-np.inf, 0.01, np.inf]
    df["score"] = pd.cut(df["eng_rate"].values, bins=bins, labels=["low < 0.01", "high > 0.01"])

    sns.countplot(df.score)
    plt.xlabel('engagement rate')
    plt.ylabel('#posts')

    plt.title(f"low vs high engagement rate")
    plt.title(f"low vs high engagement rate")
    plt.savefig(f"{saving_path}/groups_low_high.pdf")


    # print("high:", len(df["score"][df["score"] == "low < 0.01"]) / len(df["score"]))
    # print("low:", len(df["score"][df["score"] == "high > 0.01"]) / len(df["score"]))


qm_regex = re.compile(r'[\?,!.]')

def pancuations(df):
    print("Pancuation Stats")

    df['qm'] = df['post_text'].str.contains(qm_regex)
    print(len(df['qm'][df['qm'] == 1]))
    print(len(df['qm'][df['qm'] == 0]))
    print("eng_rate has qm:", np.mean(df["eng_rate"][df['qm'] == 1]))
    print("eng_rate no qm:", np.mean(df["eng_rate"][df['qm'] == 0]))

    print("abs_error has qm:", np.mean(df["abs_error"][df['qm'] == 1]))
    print("abs_error no qm:", np.mean(df["abs_error"][df['qm'] == 0]))

    # top 10%
    print("top 10% eng_rate")
    print("abs_error has qm",
          np.mean(df[df["qm"] == 1].nlargest(len(df[df["qm"] == 1]) // 10, 'eng_rate')['eng_rate']))
    print("abs_error no qm",
          np.mean(df[df["qm"] == 0].nlargest(len(df[df["qm"] == 0]) // 10, 'eng_rate')['eng_rate']))

def balance_data(df):
    rate_squared = np.power(df["eng_rate"], 2)
    sum = np.sum(rate_squared)
    p = rate_squared / sum
    df = df.reset_index()
    posts = np.random.choice(df.index.to_numpy(), size=int(len(df) * (2 / 3)), p=p.to_numpy(), replace=False)
    df = df.iloc[posts]
    return df

def main():
    df = pd.read_csv(path)
    if args.balance_data == "yes":
        df = balance_data(df)
    if stat_type == "hist" and args.by_year == "yes":
        plot_feature_histogram_by_year(df, args.feature)
    elif stat_type == "hist":
        plot_feature_histogram(df, args.feature)
    elif stat_type == "correlation":
        plot_correlations(df, args.feature, args.target)
    elif stat_type == "trendline":
        plot_trendline(df, args.feature, args.target)
    elif stat_type == "groups":
        groups(df, args.feature, args.target)
    elif stat_type == "trigger":
        trigger(df)
    elif stat_type == "pancuations":
        pancuations(df)
    else:
        print("sorry, this stat type is not supported")

if __name__ == "__main__":
    main()
