pip freeze > requirements.txtfrom os.path import isfile, join
from os import listdir
import pandas as pd
from sklearn.metrics import mean_absolute_error, recall_score, precision_score, accuracy_score, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import pearsonr, spearmanr

import argparse
parser = argparse.ArgumentParser(description='Process input')

parser.add_argument('-p', dest='path', type=str, default="regressors_out/MLP_predictions.csv", required=False)
parser.add_argument('-s', dest='saving_path', type=str, default="data_stat_out", required=False)
parser.add_argument('-r', dest='regressor', type=str, default="MLP", required=False)

args = parser.parse_args()

def regression_measures(df, regressor):
    print(f"{regressor} measures")
    df_low = df[df["eng_rate"] <= 0.01]
    df_high = df[df["eng_rate"] > 0.01]
    print("mae total:", mean_absolute_error(df["eng_rate"],  df["predictions"]))
    print("mae low eng rate:", mean_absolute_error(df_low["eng_rate"],  df_low["predictions"]))
    print("mae high eng rate:", mean_absolute_error(df_high["eng_rate"],  df_high["predictions"]))

    bins = [-np.inf, 0.01, np.inf]
    y_score = pd.cut(df["eng_rate"].values, bins=bins, labels=[0, 1])
    y_score_pred = pd.cut(df["predictions"].values, bins=bins, labels=[0, 1])

    print("recall:", recall_score(y_score, y_score_pred))
    print("precision:", precision_score(y_score, y_score_pred))
    print("accuracy:", accuracy_score(y_score, y_score_pred))
    print("f1:", f1_score(y_score, y_score_pred))

    print("Mean Values")
    print("all data")
    print("mean eng rate true:", np.mean(df["eng_rate"]))
    print("mean eng rate predicted:", np.mean(df["predictions"]))

    print("low data")
    print("mean eng rate true:", np.mean(df["eng_rate"][df["eng_rate"] <= 0.01]))
    print("mean eng rate predicted:", np.mean(df["predictions"][df["eng_rate"] <= 0.01]))

    print("high data")
    print("mean eng rate true:", np.mean(df["eng_rate"][df["eng_rate"] > 0.01]))
    print("mean eng rate predicted:", np.mean(df["predictions"][df["eng_rate"] > 0.01]))


def conf_mat(df, regressor):
    bins = [-np.inf, 0.01, np.inf]
    y_score = pd.cut(df["eng_rate"].values, bins=bins, labels=[0, 1])
    y_score_pred = pd.cut(df["predictions"].values, bins=bins, labels=[0, 1])

    labels = ['low', 'high']
    cm = confusion_matrix(y_score, y_score_pred)
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, xticklabels=labels,
                yticklabels=labels, cmap='coolwarm', annot_kws={'size': 25})
    plt.rcParams['font.size'] = 25
    plt.title(f"{regressor} confusion matrix")
    plt.xlabel('Predicted', fontsize=25)
    plt.ylabel('True', fontsize=25)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.show()


df = pd.read_csv(args.path)
regression_measures(df, args.regressor)
conf_mat(df, args.regressor)
