import pandas as pd
import numpy as np

from calc_utils import divide
from nlp_utils import stopwords
from param_config import config

def preprocess(df):
    for col in ["processd_question1", "processd_question2"]:
        df["%s_token" % col] = [[word for word in sentence.split()] for sentence in df[col].astype(str)]

def stops_ratio(q_list):
    q_stops = set(q_list).intersection(stopwords)
    return divide(len(q_stops), len(set(q_list)))

def stops_ratio_diff(stops1_ratio, stops2_ratio):
    return abs(stops1_ratio - stops2_ratio)

if __name__ == "__main__":
    df_train = pd.read_csv(config.processed_train_data_path)
    df_test = pd.read_csv(config.processed_test_data_path)
    preprocess(df_train)
    preprocess(df_test)

    df_train["stops1_ratio"] = df_train["processd_question1_token"].astype(str).apply(stops_ratio)
    df_test["stops1_ratio"] = df_test["processd_question1_token"].astype(str).apply(stops_ratio)

    df_train["stops2_ratio"] = df_train["processd_question2_token"].astype(str).apply(stops_ratio)
    df_test["stops2_ratio"] = df_test["processd_question2_token"].astype(str).apply(stops_ratio)

    df_train["stops_ratio_diff"] = df_train.apply(lambda row: stops_ratio_diff(row["stops1_ratio"], row["stops2_ratio"]), axis=1)
    df_test["stops_ratio_diff"] = df_test.apply(lambda row: stops_ratio_diff(row["stops1_ratio"], row["stops2_ratio"]), axis=1)

    feats = ["stops1_ratio", "stops2_ratio", "stops_ratio_diff"]
    df_train[feats].to_csv("{}/train_stopwords.csv".format(config.feat_folder), index=False)
    df_test[feats].to_csv("{}/test_stopwords.csv".format(config.feat_folder), index=False)