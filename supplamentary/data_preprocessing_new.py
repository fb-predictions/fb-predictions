
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from os.path import isfile, join
from json import loads as json_loads
from transformers import BertTokenizer
from datetime import datetime
import emoji
import re
from collections import Counter

from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()



#tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

mypath = "./data"


def split(df):
    df_train, df_test = train_test_split(df, test_size=0.15, random_state=1)
    df_train, df_validation = train_test_split(df_train, test_size=0.18, random_state=1)
    return df_train, df_validation, df_test


""" def scale(df_train, df_validation, df_test):
    scalar = PowerTransformer()#StandardScaler()
    scalar = scalar.fit(df_train["likes"].values.reshape(-1,1))
    df_train["scaled_likes"] = scalar.transform(df_train["likes"].values.reshape(-1,1))
    df_validation["scaled_likes"] = scalar.transform(df_validation["likes"].values.reshape(-1,1))
    df_test["scaled_likes"] = scalar.transform(df_test["likes"].values.reshape(-1,1))
    return df_train, df_validation, df_test """



emoji_dict = {
    "love": r"[♥💘💖💗💓💙💚💛❤🖤💟💞🧡💜❣🤎🤍💕😍🥰]+|(<3)+",
    "laugh": r"[😂🤣]+",
    "cry": r"[😿😢😭]+",
    "applause": r"[👏]+",
    "sad": r"[😟😞😓☹😥😔😖😩]+|(:\()+",
    "like": r"[👍]+",
    "joy": r"[😊☺😃😄😸😀😆🙃]+|(:\))+|(:D)+",
    "eye-roll": r"[🙄]+",
    "angry": r"[😠😡👿😒]+",
    "playful": r"[😉😈😜😛]+|(;\))+",
    "surprised": r"[😮]+",
    "excellent": r"[🔥💣💥]+",
    "thinking": r"[🤔]+"
}
emoji_regex = {category: re.compile(emo_group) for category, emo_group in emoji_dict.items()}
# emoji_unknown = re.compile(r"((" + r")|(".join( map(re.escape, emoji.UNICODE_EMOJI.keys()) ) + r"))+")
emoji_unknown = re.compile("[\U00010000-\U0010ffff]+", re.UNICODE)
hyperlink_regex = re.compile(r"https?://[^\s]+")
response_regex = re.compile(r'[#@][A-Za-z]*[0-9]+\s')
page_post_id_regex = re.compile(r'^#[A-Za-z]*[0-9,]+:?\s+|^[0-9,]+[:.]\s+')
hashtag_regex = re.compile(r"#([a-zA-Z_]+)")
universities_regex = re.compile(r"\b|\b".join([r"\bmit", "nyu", "ucl", "sfu", "ubc", r"bu\b"]), re.IGNORECASE)
submit_regex = re.compile(r"\nSubmitted:.+$")
zone_regxe = re.compile(r"America/|Europe/")

def post_text_proc(df):
    """
    1. Delete the post's opening number ("#bla1500")
    2. Count post's words
    3. Detect if the post is written in response to another
    4. Remove very long posts
    """
    df = df.dropna(subset=['post_text'], how='any')

    # remove page's internal id numbers
    df['post_text'] = df['post_text'].str.replace(page_post_id_regex, '').str.strip()
    df['post_text'] = df['post_text'].str.replace(submit_regex, "")

    df['words_count'] = df['post_text'].str.split().str.len()

   # drop long and short posts
   #  df = df[df["words_count"] <= 256]
    df = df.drop(df[df["words_count"] > 256].index)

    # df = df[df["words_count"] > 20]
    # df = df.drop(df[df["words_count"] < 20].index)
    #
    # df['post_text'][df["words_count"] > 256] = df['post_text'][df["words_count"] > 256].str.split().str[:100] + df['post_text'][df["words_count"] > 256].str.split().str[-150:]
    # df['post_text'][df["words_count"] > 256] = [' '.join(map(str, l)) for l in df['post_text'][df["words_count"] > 256]]

    df['char_count'] = df['post_text'].str.len()
    df['long_post'] = (df["words_count"].values > 30).astype("int64")

    df = df.dropna(subset=['post_text'], how='any')

    # response hashtags
    print("stripping response hashtags...", end="\r")
    df['is_response'] = df['post_text'].str.match(response_regex).astype("Int64")
    df['post_text'] = df['post_text'].str.replace(response_regex, "")

    # drop hashtags
    df['post_text'] = df['post_text'].str.replace(hashtag_regex, r"\1")

    df = df.dropna(subset=['post_text'], how='any')

    print("special emojis...", end="\r")
    # special emojis
    for category, reg in emoji_regex.items():
        df['post_text'] = df['post_text'].str.replace(reg, f", {category}.")

    # unknown emojiss
    df['post_text'] = df['post_text'].str.replace(emoji_unknown, "")

    # drop hyprelinks
    df['post_text'] = df['post_text'].str.replace(hyperlink_regex, "hyperlink")

    # universities
    df['post_text'] = df['post_text'].str.replace(universities_regex, "university")

    # add sentiment scores
    sentiment = df["post_text"].apply(lambda x: sid.polarity_scores(x.replace(r"[^a-zA-Z0-9.,!?\s-]", "")))
    df["neg"] = sentiment.apply(lambda x: x["neg"])
    df["pos"] = sentiment.apply(lambda x: x["pos"])
    df["neu"] = sentiment.apply(lambda x: x["neu"])



    return df


def split_time(df, tz):
    # timestamp of 24/07/2020
    max_ts = 1595546243
    max_old = 60*60*24*3

    df = df[max_ts - df["time"] > max_old]

    date = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC')

    # change time zone
    #date = date.dt.tz_localize(tz="Asia/Jerusalem", ambiguous=True, nonexistent="shift_backward")
    date = date.dt.tz_convert(tz=tz)

    #print(date.head())

    df['year'] = date.dt.year.astype('Int64')
    df['month'] = date.dt.month.astype('Int64')
    df['day'] = date.dt.dayofweek.astype('Int64')
    df['hour'] = date.dt.hour.astype('Int64')

    minute_of_day = 2*np.pi*(date.dt.hour.astype('Int64') * 60 + date.dt.minute.astype('Int64'))/1440
    df["minute_of_day"] = minute_of_day
    df["day_time_sin"] = np.sin(minute_of_day.to_numpy())/2 + 0.5
    df["day_time_cos"] = np.cos(minute_of_day.to_numpy())/2 + 0.5

    # df['time_zone'] = {"America/New_York": 0, "America/Vancouver": 1, "Europe/London": 2, "America/Los_Angeles": 3}[tz]
    df["america"] = 1 if tz in ["America/New_York", "America/Los_Angeles"] else 0
    df["uk"] = 1 if tz in ["Europe/London"] else 0
    df["canada"] = 1 if tz in ["America/Vancouver"] else 0

    df["time_zone"] = tz
    df["time_zone"] = df["time_zone"].str.replace(zone_regxe, "")

    for day in range(7):
        df[f"day_{day+1}"] = (df["day"].values == day).astype("int64")

    for month in range(12):
        df[f"month_{month+1}"] = (df["month"].values == month+1).astype("int64")

    return df

def split_time_new_data(df):
    # timestamp of 24/07/2020
    max_ts = 1595546243
    max_old = 60*60*24*3

    date = pd.to_datetime(df['time_posted'], dayfirst=True)

    df['year'] = date.dt.year.astype('Int64')
    df['month'] = date.dt.month.astype('Int64')
    df['day'] = date.dt.dayofweek.astype('Int64')
    df['hour'] = date.dt.hour.astype('Int64')

    minute_of_day = 2*np.pi*(date.dt.hour.astype('Int64') * 60 + date.dt.minute.astype('Int64'))/1440
    df["minute_of_day"] = minute_of_day
    df["day_time_sin"] = np.sin(minute_of_day.to_numpy())/2 + 0.5
    df["day_time_cos"] = np.cos(minute_of_day.to_numpy())/2 + 0.5

    # df['time_zone'] = {"America/New_York": 0, "America/Vancouver": 1, "Europe/London": 2, "America/Los_Angeles": 3}[tz]
    df["america"] = 0
    df["uk"] = 1
    df["canada"] = 0

    df["time_zone"] = "London"

    for day in range(7):
        df[f"day_{day+1}"] = (df["day"].values == day).astype("int64")

    for month in range(12):
        df[f"month_{month+1}"] = (df["month"].values == month+1).astype("int64")

    return df

def drop_unknown(df):
    indices = []
    #sume = 0
    for idx, row in df.iterrows():
        tokens = tokenizer.tokenize(row["post_text"])
        if "[UNK]" in tokens:
            unk = sum([1 for t in tokens if t == "[UNK]"])
            perc = unk / row["words_count"]
            if perc >= 0.1:
                #sume += 1
                indices.append(idx)
                #print(f"{unk}/{row['words_count']} ({perc}) - likes: {row['likes']}")
    #print(f"deleted: {len(indices)}")

    df = df.drop(indices)
    return df


def calc_engagement(df, followers):
    # median weekly growth according to https://socialhospitality.com/2013/03/study-average-growth-of-facebook-fan-pages/
    # we know for a fact that mit had 29275 followers in 2/12/2018, so it seems like this estimate is still correct
    # weekly_growth = 1.0064
    #
    # # date of page followers counting
    # dt_max = datetime.fromtimestamp(df["time"].iloc[0])
    #
    # posts_days = pd.to_datetime(df['time'], unit='s')#.dt.tz_localize('UTC').dt.tz_convert('Asia/Jerusalem')
    # deltas_days = posts_days.sub(dt_max).dt.days * (-1)
    # delta_weeks = deltas_days / 7
    # fitted_followers = followers*(0.9/np.power(weekly_growth, delta_weeks) + 0.1)
    # # df["fitted_followers"] = fitted_followers
    # df["fitted_followers"] = fitted_followers * np.power(0.99, df["year"] - birth_year)
    #
    # reactions = df["total_reactions"].values
    #
    # if reactions.dtype == "object":
    #     df["total_reactions"] = df["total_reactions"].str.replace("K", "e3").astype("float")
    #     df["total_reactions"] = df["total_reactions"].astype("int64")
    #     reactions = df[['total_reactions', 'likes']].values.max(1)
    #
    # total_eng = reactions + df["comments"].values + df["shares"].values
    # df["total_eng"] = total_eng
    # factors = total_eng/150
    # factors[factors > 1.5] = 1.5
    # factors += 0.5
    #
    # # df["eng_rate"] = 100 * total_eng / fitted_followers
    # df["eng_rate"] = factors * total_eng / fitted_followers
    #
    # df["score"] = pd.cut(df["eng_rate"].values, bins=[0, 0.09, 0.9, np.inf], labels=[0, 1, 2], include_lowest=True, right=False)

    weekly_growth = 1.0064

    # date of page followers counting
    dt_max = datetime.fromtimestamp(df["time"].iloc[0])

    posts_days = pd.to_datetime(df['time'], unit='s')#.dt.tz_localize('UTC').dt.tz_convert('Asia/Jerusalem')
    deltas_days = posts_days.sub(dt_max).dt.days * (-1)
    delta_weeks = deltas_days / 7
    fitted_followers = followers / (np.power(weekly_growth, delta_weeks))
    # df["fitted_followers"] = fitted_followers
    df["fitted_followers"] = fitted_followers #* np.power(0.99, df["year"] - birth_year)

    reactions = df["total_reactions"].values

    if reactions.dtype == "object":
        df["total_reactions"] = df["total_reactions"].str.replace("K", "e3").astype("float")
        df["total_reactions"] = df["total_reactions"].astype("int64")
        reactions = df[['total_reactions', 'likes']].values.max(1)

    total_eng = reactions + df["comments"].values + df["shares"].values
    df["total_eng"] = total_eng
   # factors = total_eng/150
    #factors[factors > 1.5] = 1.5
    #factors += 0.5

    # df["eng_rate"] = 100 * total_eng / fitted_followers
    df["eng_rate"] = total_eng / fitted_followers

    # df["score"] = pd.qcut(df["eng_rate"].values, q=[0, 0.25, 0.5, 1], labels=["__label__0", "__label__1", "__label__2"])
    # df["score"] = pd.qcut(df["eng_rate"].values, q=[0, 0.5, 0.75, 1], labels=[0, 1, 2])

    return df


def calc_engagement_new_data(df, followers):
    weekly_growth = 1.0064

    groups = ["Edifess", "LeedsFess", "Linconfess", "Bristruths", "Newfess"]
    df_new = pd.DataFrame()
    for group in groups:
        df_group = df[df["name"] == group]
        dt_max = datetime.strptime(df_group["time_posted"].iloc[0], "%Y-%m-%d %H:%M:%S")

        posts_days = pd.to_datetime(df_group["time_posted"], dayfirst=True)  # .dt.tz_localize('UTC').dt.tz_convert('Asia/Jerusalem')
        deltas_days = posts_days.sub(dt_max).dt.days * (-1)
        delta_weeks = deltas_days / 7
        fitted_followers = followers[group] / (np.power(weekly_growth, delta_weeks))
        # df["fitted_followers"] = fitted_followers
        df_group["fitted_followers"] = fitted_followers  # * np.power(0.99, df["year"] - birth_year)


        # factors = total_eng/150
        # factors[factors > 1.5] = 1.5
        # factors += 0.5

        # df["eng_rate"] = 100 * total_eng / fitted_followers
        df_group["eng_rate"] = df_group["total_eng"] / fitted_followers

        # df_group["score"] = pd.qcut(df_group["eng_rate"].values, q=[0, 0.5, 0.75, 1], labels=[0, 1, 2])
        df_new = df_new.append(df_group)
    return df_new

# def scale_features(df_train, df_val, df_test, features):
#     robust_scaler_char = RobustScaler().fit(df_train["char_count"].values.reshape(-1, 1))
#     robust_scaler_words = RobustScaler().fit(df_train["words_count"].values.reshape(-1, 1))
#     for df in [df_train, df_val, df_test]:
#         df["words_count_scaled"] = robust_scaler_words.transform(df["words_count"].values.reshape(-1, 1))
#         df["char_count_scaled"] = robust_scaler_char.transform(df["char_count"].values.reshape(-1, 1))
#     return df_train, df_val, df_test

def scale_features(df_train, df_val, df_test, features):
    scaler_char = RobustScaler().fit(df_train["char_count"].values.reshape(-1, 1))
    scaler_words = RobustScaler().fit(df_train["words_count"].values.reshape(-1, 1))
    scaler_followers = RobustScaler().fit(df_train["fitted_followers"].values.reshape(-1, 1))
    for df in [df_train, df_val, df_test]:
        df["words_count_scaled"] = scaler_words.transform(df["words_count"].values.reshape(-1, 1))
        df["char_count_scaled"] = scaler_char.transform(df["char_count"].values.reshape(-1, 1))
        df["fitted_followers_scaled"] = scaler_followers.transform(df["fitted_followers"].values.reshape(-1, 1))
    return df_train, df_val, df_test


def scale_targets(df_train, df_val, df_test):
    standard_scaler = StandardScaler().fit(df_train["eng_rate"].values.reshape(-1, 1))
    for df in [df_train, df_val, df_test]:
        df["eng_rate_scaled"] = standard_scaler.transform(df["eng_rate"].values.reshape(-1, 1))
        # df["eng_rate_scaled"] = 1 / (1 + np.exp(-df["eng_rate_scaled"]))
    return df_train, df_val, df_test


def collect_emojis(df):
    all_emojis = []
    for idx, row in df.iterrows():
        row_emojis = ':'.join(c for c in str(row.post_text) if c in emoji.UNICODE_EMOJI)
        if len(row_emojis) > 0:
            all_emojis.append(row_emojis)
    all_emojis_str = ':'.join(str(elem) for elem in all_emojis)

    return all_emojis_str


def emojis_statistics(all_emojis_list):
    all_emojis_str = ':'.join(all_emojis_list)

    all_emojis_list = re.split(':', all_emojis_str)

    c = Counter(all_emojis_list)
    print(c)

def balance_data(df):
    rate_squared = np.power(df["eng_rate"], 2)
    sum = np.sum(rate_squared)
    p = rate_squared / sum
    df = df.reset_index()
    posts = np.random.choice(df.index.to_numpy(), size=int(len(df) * (2 / 3)), p=p.to_numpy(), replace=False)
    df = df.iloc[posts]
    return df



def main():

    pd.options.mode.chained_assignment = None

    tz_handler = None
    with open("../raw_data/timezones.json", "r") as lf:
        tz_handler = json_loads(lf.read())

    followers = None
    with open("../raw_data/followers.json", "r") as lf:
        followers = json_loads(lf.read())

    birth_year = None
    with open("../raw_data/birth_year.json", "r") as lf:
        birth_year = json_loads(lf.read())

    all_train, all_validation, all_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(join(mypath, "posts_1650.csv"))

    df = df.rename(columns={'total_reactions_count': 'total_eng', 'text': 'post_text'})
    df = df.dropna(subset=['post_text', 'total_eng', 'time_posted'], how='any')
    df = split_time_new_data(df)
    df = calc_engagement_new_data(df, followers)
    df = post_text_proc(df)
    df.replace('', np.nan, inplace=True)
    df = df.dropna(subset=['post_text', 'total_eng', 'time_posted'], how='any')
    df = drop_unknown(df)
    df = df.dropna(subset=['eng_rate'], how='any')
    df = df.drop(["time_posted", "taken_at"], axis=1)
    # df = balance_data(df)

    for group in ['Bristruths', 'Edifess', 'LeedsFess', 'Linconfess', 'Newfess']:
        df_group = df[df['name'] == group]
        df_train, df_validation, df_test = split(df_group)
        all_train = all_train.append(df_train)
        all_validation = all_validation.append(df_validation)
        all_test = all_test.append(df_test)


    f_count = 0
    all_emojis_list = []
    all_groups = followers.keys()

    total_length = 0
    for current_group in all_groups:
        if current_group in ['beaverconfessions', 'Bristruths', 'Edifess', 'LeedsFess', 'Linconfess', 'Newfess']:
           continue
        df = pd.read_csv(join(mypath, current_group) + ".csv")
        df_reactions = pd.read_csv(join(mypath, current_group) + "_reactions.csv")
        # total_length += len(df)
        df["total_reactions"] = df_reactions["total_reactions"]
        df["top_reactions"] = df_reactions["top_reactions"]

        f_count += 1

        #emojis_str = collect_emojis(df)
        #all_emojis_list.append(emojis_str)

        print(f"Current: {f_count}/{len(followers.keys())} - {current_group}")

        df = df[pd.isnull(df["image"])]
        df = df[pd.isnull(df["video"])]
        df = df[pd.isnull(df["link"])]
        df = df[pd.isnull(df["shared_text"])]

        df = df.drop(["post_id", "text", "shared_text", "image", "link", "video", "post_url"], axis=1)

        df = df.dropna(subset=['post_text', 'likes', 'comments', 'shares', 'total_reactions'], how='any')

        # correct the time zone and handle the time of the page
        tz = tz_handler[current_group]
        df = split_time(df, tz)
        df = df.dropna(subset=['month', 'day', 'hour'], how='any')

        df = calc_engagement(df, followers[current_group])

        df = post_text_proc(df)
        df = drop_unknown(df)
        # df = balance_data(df)
        df_train, df_validation, df_test = split(df)

        all_train = all_train.append(df_train)
        all_validation = all_validation.append(df_validation)
        all_test = all_test.append(df_test)

        # print(f"{current_group}")
        # print(df["total_eng"].describe())
        # print("*************")



    #emojis_statistics(all_emojis_list)
    features_scale = ["words_count", "char_count"]

    # for feature in features_scale:
    #     plot("Train", all_train, feature)



    all_train, all_validation, all_test = scale_features(all_train, all_validation, all_test, features_scale)
    # all_train, all_validation, all_test = scale_targets(all_train, all_validation, all_test)





    all_train["score"] = pd.cut(all_train["total_eng"].values, bins=[-np.inf, 50, 200, np.inf], labels=[0, 1, 2])
    all_validation["score"] = pd.cut(all_validation["total_eng"].values, bins=[-np.inf, 50, 200, np.inf], labels=[0, 1, 2])
    all_test["score"] = pd.cut(all_test["total_eng"].values, bins=[-np.inf, 50, 200, np.inf], labels=[0, 1, 2])
    all_train = all_train.drop(['time', 'likes', 'comments', 'shares', 'total_reactions', 'top_reactions', 'name'], axis=1)


    all_train = all_train.dropna(subset=['post_text', 'total_eng', 'eng_rate'])
    all_validation = all_validation.dropna(subset=['post_text', 'total_eng', 'eng_rate'])
    all_test = all_test.dropna(subset=['post_text', 'total_eng', 'eng_rate'])

    # all_train = balance_data(all_train)


    all_train.to_csv(f"../data/all_train.csv")
    all_validation.to_csv(f"../data/all_validation.csv")
    all_test.to_csv(f"../data/all_test.csv")


if __name__ == "__main__":
    main()


