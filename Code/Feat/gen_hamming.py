import numpy as np
import pandas as pd
import distance

from calc_utils import divide
from param_config import config


def preprocess(df):
    for col in ["processd_question1", "processd_question2"]:
        df[col] = [[word for word in sentence.split()] for sentence in df[col].astype(str)]

def words_hamming(q1_list, q2_list):
    return divide(sum(1 for i in zip(q1_list, q2_list) if i[0] == i[1]), max(len(q1_list), len(q2_list)))

if __name__ == "__main__":
    df_train = pd.read_csv(config.processed_train_data_path)
    df_test = pd.read_csv(config.processed_test_data_path)
    preprocess(df_train)
    preprocess(df_test)

    df_train["words_hamming"] = df_train.apply(lambda row: words_hamming(row["processd_question1"], row["processd_question2"]), axis=1)
    df_test["words_hamming"] = df_test.apply(lambda row: words_hamming(row["processd_question1"], row["processd_question2"]), axis=1)

    feats = ["words_hamming"]
    df_train[feats].to_csv("{}/train_hamming.csv".format(config.feat_folder), index=False)
    df_test[feats].to_csv("{}/test_hamming.csv".format(config.feat_folder), index=False)