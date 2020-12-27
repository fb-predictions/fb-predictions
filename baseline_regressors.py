import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.base import TransformerMixin
import numpy as np
from sklearn_features.transformers import DataFrameSelector
from sklearn.tree import DecisionTreeRegressor

data_path = "./data"
relevant_features = ['words_count_scaled', 'char_count_scaled', 'is_response', 'long_post', 'day_time_sin', 'day_time_cos', 'neg', 'pos', 'neu',  'fitted_followers_scaled']
all_features = ['post_text', 'words_count_scaled', 'char_count_scaled', 'is_response', 'long_post', 'day_time_sin', 'day_time_cos', 'neg', 'pos', 'neu', 'fitted_followers_scaled']

import argparse
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('-p', dest='path', type=str, default="processed_data", required=False)
parser.add_argument('-s', dest='saving_path', type=str, default="regressors_out", required=False)
parser.add_argument('-r', dest='regressor', type=str, default="MLP", required=False)
parser.add_argument('-b', dest='balance_data', type=str, default="no", required=False)

args = parser.parse_args()

def balance_data(df):
    rate_squared = np.power(df["eng_rate"], 2)
    sum = np.sum(rate_squared)
    p = rate_squared / sum
    df = df.reset_index()
    posts = np.random.choice(df.index.to_numpy(), size=int(len(df) * (2 / 3)), p=p.to_numpy(), replace=False)
    df = df.iloc[posts]
    return df


df_train = pd.read_csv(f"{args.path}/all_train.csv")
if args.balance_data == "yes":
    df_train = balance_data(df_train)

df_train = df_train[df_train["year"] > 2017]
df_train = df_train.dropna(subset=relevant_features)
df_train = df_train.dropna(subset=["post_text", "eng_rate"])
df_train = df_train.reset_index()
X_train = df_train[all_features]
y_train = df_train["eng_rate"]




df_val = pd.read_csv(f"{args.path}/all_validation.csv")
df_val = df_val[df_val["year"] > 2017]
df_val = df_val.dropna(subset=relevant_features)
df_val = df_val.dropna(subset=["post_text", "eng_rate"])
df_val = df_val.reset_index()
X_val = df_val[all_features]
y_val = df_val["eng_rate"]
y_val = y_val.to_numpy()

regressors = {'Linear': LinearRegression(), 'MLP': MLPRegressor(), 'Decision_Tree': DecisionTreeRegressor()}

mean = np.mean(y_train)
std = np.std(y_train)

def scale(targets):

    targets = (targets - mean) / std
    return targets

def unscale(targets):

    targets = targets * std + mean
    return targets

class DenseTransformer(TransformerMixin):

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


reg = regressors[args.regressor]
print(f"Training {reg}...")

pipeline = Pipeline([
    ('union', FeatureUnion(
        transformer_list=[
            ('non_text', DataFrameSelector(relevant_features)),
            #
            ('text', Pipeline([
                ('text_data', DataFrameSelector("post_text")),
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
            ])),
        ],
        transformer_weights={
            'text': 0.5,
            'non_text': 0.5,
        },
    )),
    ('reg', reg),
])

pipeline.fit(X_train, scale(y_train))

predicted = unscale(pipeline.predict(X_val))

df_val.to_csv(f"{args.saving_path}/{reg}_predictions.csv")