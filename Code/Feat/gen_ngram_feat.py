import pandas as pd
import numpy as np

from calc_utils import divide
from ngram import get_bigram, get_trigram
from nlp_utils import stopwords
from param_config import config

def preprocess(df):
    for col in ["processd_question1", "processd_question2"]:
        df["%s_token" % col] = [[word for word in sentence.split()] for sentence in df[col].astype(str)]

def ngram_jaccard(q1_ngram, q2_ngram):
    q1_ngram, q2_ngram = set(q1_ngram), set(q2_ngram)
    shared_ngram = q1_ngram.intersection(q2_ngram)
    return divide(len(shared_ngram), len(q1_ngram.union(q2_ngram)))

def ngram_jaccard_sum(q1_ngram, q2_ngram):
    q1_ngram, q2_ngram = set(q1_ngram), set(q2_ngram)
    shared_ngram = q1_ngram.intersection(q2_ngram)
    return divide(len(shared_ngram), len(q1_ngram) + len(q2_ngram))

def ngram_jaccard_max(q1_ngram, q2_ngram):
    q1_ngram, q2_ngram = set(q1_ngram), set(q2_ngram)
    shared_ngram = q1_ngram.intersection(q2_ngram)
    return divide(len(shared_ngram), max(len(q1_ngram), len(q2_ngram)))

if __name__ == "__main__":

    df_train = pd.read_csv(config.processed_train_data_path)
    df_test = pd.read_csv(config.processed_test_data_path)
    preprocess(df_train)
    preprocess(df_test)

    df_train["question1_bigram"] = df_train["processd_question1_token"].apply(get_bigram)
    df_test["question1_bigram"] = df_test["processd_question1_token"].apply(get_bigram)

    df_train["question2_bigram"] = df_train["processd_question2_token"].apply(get_bigram)
    df_test["question2_bigram"] = df_test["processd_question2_token"].apply(get_bigram)

    df_train["question1_trigram"] = df_train["processd_question1_token"].apply(get_trigram)
    df_test["question1_trigram"] = df_test["processd_question1_token"].apply(get_trigram)

    df_train["question2_trigram"] = df_train["processd_question2_token"].apply(get_trigram)
    df_test["question2_trigram"] = df_test["processd_question2_token"].apply(get_trigram)

    #df_train["bigram_jaccard"] = df_train.apply(lambda row: ngram_jaccard(row["question1_bigram"], row["question2_bigram"]), axis=1)
    #df_test["bigram_jaccard"] = df_test.apply(lambda row: ngram_jaccard(row["question1_bigram"], row["question2_bigram"]), axis=1)

    df_train["bigram_jaccard_sum"] = df_train.apply(lambda row: ngram_jaccard_sum(row["question1_bigram"], row["question2_bigram"]), axis=1)
    df_test["bigram_jaccard_sum"] = df_test.apply(lambda row: ngram_jaccard_sum(row["question1_bigram"], row["question2_bigram"]), axis=1)

    #df_train["bigram_jaccard_max"] = df_train.apply(lambda row: ngram_jaccard_max(row["question1_bigram"], row["question2_bigram"]), axis=1)
    #df_test["bigram_jaccard_max"] = df_test.apply(lambda row: ngram_jaccard_max(row["question1_bigram"], row["question2_bigram"]), axis=1)

    #df_train["trigram_jaccard"] = df_train.apply(lambda row: ngram_jaccard(row["question1_trigram"], row["question2_trigram"]), axis=1)
    #df_test["trigram_jaccard"] = df_test.apply(lambda row: ngram_jaccard(row["question1_trigram"], row["question2_trigram"]), axis=1)

    df_train["trigram_jaccard_sum"] = df_train.apply(lambda row: ngram_jaccard_sum(row["question1_trigram"], row["question2_trigram"]), axis=1)
    df_test["trigram_jaccard_sum"] = df_test.apply(lambda row: ngram_jaccard_sum(row["question1_trigram"], row["question2_trigram"]), axis=1)

    #df_train["trigram_jaccard_max"] = df_train.apply(lambda row: ngram_jaccard_max(row["question1_trigram"], row["question2_trigram"]), axis=1)
    #df_test["trigram_jaccard_max"] = df_test.apply(lambda row: ngram_jaccard_max(row["question1_trigram"], row["question2_trigram"]), axis=1)

    #feats = ["bigram_jaccard", "bigram_jaccard_sum", "bigram_jaccard_max", "trigram_jaccard", "trigram_jaccard_sum", "trigram_jaccard_max"]
    feats = ["bigram_jaccard_sum", "trigram_jaccard_sum"]
    df_train[feats].to_csv("{}/train_ngram.csv".format(config.feat_folder), index=False)
    df_test[feats].to_csv("{}/test_ngram.csv".format(config.feat_folder), index=False)

