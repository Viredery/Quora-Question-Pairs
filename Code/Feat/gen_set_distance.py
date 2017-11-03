import numpy as np
import pandas as pd
import distance

from nlp_utils import stopwords
from calc_utils import divide
from param_config import config


def preprocess(df):
    for col in ["processd_question1", "processd_question2"]:
        df[col] = [set([word for word in sentence.split() if word not in stopwords]) for sentence in df[col].astype(str)]

def gen_shared_words_len(q1_set, q2_set):
    return len(set(q1_set).intersection(q2_set))

def gen_shared_words_len_normalized_max(q1_set, q2_set):
    if type(q2_set) == str:
        q2_set = set([])
    return divide(len(set(q1_set).intersection(q2_set)), max(len(set(q1_set)), len(set(q2_set))))

def gen_set_intersection(q1_set, q2_set):
    return len(set(q1_set).intersection(q2_set)) * 2.0 / (len(set(q1_set)) + len(set(q2_set)))

def gen_str_jaccard(q1_set, q2_set):
    wic = set(q1_set).intersection(q2_set)
    uw = set(q1_set).union(q2_set)
    return divide(len(wic), len(uw))

def gen_unique_words_len(q1_set, q2_set):
    return len(set(q1_set).union(q2_set))

if __name__ == '__main__':

    df_train = pd.read_csv(config.processed_train_data_path)
    df_test = pd.read_csv(config.processed_test_data_path)
    preprocess(df_train)
    preprocess(df_test)

    df_train["shared_words_len"] = df_train.astype(str).apply(lambda row: gen_shared_words_len(row['processd_question1'], row['processd_question2']), axis=1)
    df_test["shared_words_len"] = df_test.astype(str).apply(lambda row: gen_shared_words_len(row['processd_question1'], row['processd_question2']), axis=1)

    df_train["shared_words_len_normalized_max"] = df_train.astype(str).apply(lambda row: gen_shared_words_len_normalized_max(row['processd_question1'], row['processd_question2']), axis=1)
    df_test["shared_words_len_normalized_max"] = df_test.astype(str).apply(lambda row: gen_shared_words_len_normalized_max(row['processd_question1'], row['processd_question2']), axis=1)

    df_train["shared_words_len_normalized_sum"] = df_train.astype(str).apply(lambda row: gen_shared_words_len_normalized_sum(row['processd_question1'], row['processd_question2']), axis=1)
    df_test["shared_words_len_normalized_sum"] = df_test.astype(str).apply(lambda row: gen_shared_words_len_normalized_sum(row['processd_question1'], row['processd_question2']), axis=1)

    df_train["interaction"] = df_train.astype(str).apply(lambda row: gen_set_intersection(row['processd_question1'], row['processd_question2']), axis=1)
    df_test["interaction"] = df_test.astype(str).apply(lambda row: gen_set_intersection(row['processd_question1'], row['processd_question2']), axis=1)

    df_train["jaccard"] = df_train.astype(str).apply(lambda row: gen_str_jaccard(row['processd_question1'], row['processd_question2']), axis=1)
    df_test["jaccard"] = df_test.astype(str).apply(lambda row: gen_str_jaccard(row['processd_question1'], row['processd_question2']), axis=1)

    df_train["jaccard_sqrt"] = np.sqrt(df_train["jaccard"])
    df_test["jaccard_sqrt"] = np.sqrt(df_test["jaccard"])

    df_train["unique_words_len"] = df_train.astype(str).apply(lambda row: gen_unique_words_len(row['processd_question1'], row['processd_question2']), axis=1)
    df_test["unique_words_len"] = df_test.astype(str).apply(lambda row: gen_unique_words_len(row['processd_question1'], row['processd_question2']), axis=1)



    feats = ["shared_words_len", "shared_words_len_normalized_max", "shared_words_len_normalized_sum", "interaction", "jaccard", "jaccard_sqrt", "unique_words_len"]
    df_train[feats].to_csv("{}/train_set_distance.csv".format(config.feat_folder), index=False)
    df_test[feats].to_csv("{}/test_set_distance.csv".format(config.feat_folder), index=False)
