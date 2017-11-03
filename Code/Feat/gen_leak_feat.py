import pandas as pd
import numpy as np
from collections import defaultdict

from calc_utils import divide_df
from param_config import config


def gen_q_pos_freq(question, dict_to_apply):
    try:
        return dict_to_apply[question]
    except KeyError:
        return 0

def gen_freq_sum(freq1, freq2):
    return freq1 + freq2

def gen_freq_prod(freq1, freq2):
    return freq1 * freq2

def gen_freq_diff(freq1, freq2):
    return abs(1.0 * (freq1 - freq2))

def gen_max_freq(freq1, freq2):
    return 1.0 * max(freq1, freq2)

def gen_q1_q2_intersect(row):
    return(len(set(q_network_dict[row["question1"]]).intersection(set(q_network_dict[row["question2"]]))))

if __name__ == "__main__":
    df_train = pd.read_csv(config.original_train_data_path)
    df_test = pd.read_csv(config.original_test_data_path)

    ques = pd.concat([df_train[["question1", "question2"]], df_test[["question1", "question2"]]], axis=0).reset_index(drop="index")
    q1_freq_dict = ques.question1.value_counts().to_dict()
    q2_freq_dict = ques.question2.value_counts().to_dict()

    df_train["q1_pos1_freq"] = df_train.question1.map(lambda x: gen_q_pos_freq(x, q1_freq_dict))
    df_test["q1_pos1_freq"] = df_test.question1.map(lambda x: gen_q_pos_freq(x, q1_freq_dict))

    df_train["q2_pos2_freq"] = df_train.question2.map(lambda x: gen_q_pos_freq(x, q2_freq_dict))
    df_test["q2_pos2_freq"] = df_test.question2.map(lambda x: gen_q_pos_freq(x, q2_freq_dict))

    df_train["q1_freq"] = df_train.question1.map(lambda x: gen_q_pos_freq(x, q1_freq_dict) + gen_q_pos_freq(x, q2_freq_dict))
    df_test["q1_freq"] = df_test.question1.map(lambda x: gen_q_pos_freq(x, q1_freq_dict) + gen_q_pos_freq(x, q2_freq_dict))

    df_train["q2_freq"] = df_train.question2.map(lambda x: gen_q_pos_freq(x, q1_freq_dict) + gen_q_pos_freq(x, q2_freq_dict))
    df_test["q2_freq"] = df_test.question2.map(lambda x: gen_q_pos_freq(x, q1_freq_dict) + gen_q_pos_freq(x, q2_freq_dict))

    df_train["freq_sum"] = df_train.apply(lambda row: gen_freq_sum(row["q1_freq"], row["q2_freq"]), axis=1)
    df_test["freq_sum"] = df_test.apply(lambda row: gen_freq_sum(row["q1_freq"], row["q2_freq"]), axis=1)

    df_train["freq_prod"] = df_train.apply(lambda row: gen_freq_prod(row["q1_freq"], row["q2_freq"]), axis=1)
    df_test["freq_prod"] = df_test.apply(lambda row: gen_freq_prod(row["q1_freq"], row["q2_freq"]), axis=1)

    df_train["freq_diff"] = df_train.apply(lambda row: gen_freq_diff(row["q1_freq"], row["q2_freq"]), axis=1)
    df_test["freq_diff"] = df_test.apply(lambda row: gen_freq_diff(row["q1_freq"], row["q2_freq"]), axis=1)

    df_train["max_freq"] = df_train.apply(lambda row: gen_max_freq(row["q1_freq"], row["q2_freq"]), axis=1)
    df_test["max_freq"] = df_test.apply(lambda row: gen_max_freq(row["q1_freq"], row["q2_freq"]), axis=1)

    df_train["freq_diff_normalized"] = divide_df(df_train, "freq_diff", "max_freq")
    df_test["freq_diff_normalized"] = divide_df(df_test, "freq_diff", "max_freq")

    df_train["exactly_identical"] = (df_train["question1"] == df_train["question2"]).astype(int)
    df_test["exactly_identical"] = (df_test["question1"] == df_test["question2"]).astype(int)

    q_network_dict = defaultdict(set)
    for i in range(ques.shape[0]):
        q_network_dict[ques.question1[i]].add(ques.question2[i])
        q_network_dict[ques.question2[i]].add(ques.question1[i])

    df_train["q1_q2_intersect"] = df_train.apply(gen_q1_q2_intersect, axis=1)
    df_test["q1_q2_intersect"] = df_test.apply(gen_q1_q2_intersect, axis=1)



    feats = ["q1_freq", "q2_freq", "q1_pos1_freq", "q2_pos2_freq", "freq_sum", "freq_prod", "freq_diff", "max_freq", "freq_diff_normalized", "exactly_identical", "q1_q2_intersect"]
    df_train[feats].to_csv("{}/train_leak_feat.csv".format(config.feat_folder), index=False)
    df_test[feats].to_csv("{}/test_leak_feat.csv".format(config.feat_folder), index=False)

