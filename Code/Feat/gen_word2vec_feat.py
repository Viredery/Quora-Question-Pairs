import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import euclidean_distances
from gensim.models import Word2Vec
from pyemd import emd

from calc_utils import divide
from nlp_utils import stopwords
from param_config import config

CONFIG_VEC_SIZE = 200

def get_token_without_stopword(df_train, df_test):
    for col in ["processd_question1", "processd_question2"]:
        df_train["token_%s" % col] = [[word for word in line.split() if word not in stopwords] for line in df_train[col].astype(str)]
        df_test["token_%s" % col] = [[word for word in line.split() if word not in stopwords] for line in df_test[col].astype(str)]

def get_corpus(df_train, df_test):
    corpus = []
    for col in ["processd_question1", "processd_question2"]:
        for sentence in df_train[col].astype(str).iteritems():
            word_list = sentence[1].split()
            corpus.append(word_list)
        for sentence in df_test[col].astype(str).iteritems():
            word_list = sentence[1].split()
            corpus.append(word_list)
    return corpus

def gen_word_importance(model, tokens1, tokens2):
    prev_q1_len = len(tokens1)
    prev_q2_len = len(tokens2)
    q1_len = len([token for token in tokens1 if token in model])
    q2_len = len([token for token in tokens2 if token in model])
    return divide(q1_len + q2_len, prev_q1_len + prev_q2_len)

def gen_n_similarity(model, tokens1, tokens2):
    valid_q1_list = [token for token in tokens1 if token in model]
    valid_q2_list = [token for token in tokens2 if token in model]
    if len(valid_q1_list) > 0 and len(valid_q2_list) > 0:
        return model.n_similarity(valid_q1_list, valid_q2_list)
    else:
        return -1

def gen_n_similarity_imp(row):
    return row["n_similarity"] * row["word_importance"]

def get_centroid_vec(model, tokens):
    valid_list = [token for token in tokens if token in model]
    centroid = np.zeros(model.vector_size)
    for token in valid_list:
        centroid += model[token]
    if len(valid_list) > 0:
        centroid /= float(len(valid_list))
    return centroid

def gen_centroid_rmse(model, tokens1, tokens2):
    centroid_vec1 = get_centroid_vec(model, tokens1)
    centroid_vec2 = get_centroid_vec(model, tokens2)
    vdiff = centroid_vec1 - centroid_vec2
    return np.sqrt(np.mean(vdiff ** 2))

def gen_centroid_rmse_imp(row):
    return row["centroid_rmse"] * row["word_importance"]

def gen_wmd(model, s1, s2):
    return model.wmdistance(s1, s2)

def gen_wmd_imp(row):
    return row["word_mover_distance"] * row["word_importance"]


if __name__ == "__main__":

    df_train = pd.read_csv(config.processed_train_data_path)
    df_test = pd.read_csv(config.processed_test_data_path)

    corpus = get_corpus(df_train, df_test)
    model = Word2Vec(corpus, size=CONFIG_VEC_SIZE, window=5, min_count=3)

    get_token_without_stopword(df_train, df_test)

    df_train["word_importance"] = df_train.apply(lambda row: gen_word_importance(model, row["token_processd_question1"], row["token_processd_question2"]), axis=1)
    df_test["word_importance"] = df_test.apply(lambda row: gen_word_importance(model, row["token_processd_question1"], row["token_processd_question2"]), axis=1)

    df_train["n_similarity"] = df_train.apply(lambda row: gen_n_similarity(model, row["token_processd_question1"], row["token_processd_question2"]), axis=1)
    df_test["n_similarity"] = df_test.apply(lambda row: gen_n_similarity(model, row["token_processd_question1"], row["token_processd_question2"]), axis=1)

    df_train["n_similarity_imp"] = df_train.apply(gen_n_similarity_imp, axis=1)
    df_test["n_similarity_imp"] = df_test.apply(gen_n_similarity_imp, axis=1)

    df_train["centroid_rmse"] = df_train.apply(lambda row: gen_centroid_rmse(model, row["token_processd_question1"], row["token_processd_question2"]), axis=1)
    df_test["centroid_rmse"] = df_test.apply(lambda row: gen_centroid_rmse(model, row["token_processd_question1"], row["token_processd_question2"]), axis=1)

    df_train["centroid_rmse_imp"] = df_train.apply(gen_centroid_rmse_imp, axis=1)
    df_test["centroid_rmse_imp"] = df_test.apply(gen_centroid_rmse_imp, axis=1)

    df_train["word_mover_distance"] = df_train.apply(lambda row: gen_wmd(model, row["token_processd_question1"], row["token_processd_question2"]), axis=1)
    df_test["word_mover_distance"] = df_test.apply(lambda row: gen_wmd(model, row["token_processd_question1"], row["token_processd_question2"]), axis=1)

    df_train["word_mover_distance_imp"] = df_train.apply(gen_wmd_imp, axis=1)
    df_test["word_mover_distance_imp"] = df_test.apply(gen_wmd_imp, axis=1)

    feats = ["word_importance", "n_similarity", "n_similarity_imp", "centroid_rmse", "centroid_rmse_imp", "word_mover_distance", "word_mover_distance_imp"]
    df_train[feats].to_csv("{}/train_word2vec_feat.csv".format(config.feat_folder), index=False)
    df_test[feats].to_csv("{}/test_word2vec_feat.csv".format(config.feat_folder), index=False)