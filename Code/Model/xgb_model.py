# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import xgboost as xgb
import datetime
import operator
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from collections import Counter

from param_config import config

ROUNDS = 2500

print("Started")
np.random.seed(config.random_seed)

'''
def imbalance_logloss(y_pred, dtrain):
    from sklearn.preprocessing import LabelBinarizer
    eps=1e-15
    y_true = dtrain.get_label()
    lb = LabelBinarizer()
    lb.fit(y_true)
    transformed_labels = lb.transform(y_true)
    transformed_labels = np.append((1 - transformed_labels) * 1.309028344, transformed_labels * 0.472001959, axis=1)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    if y_pred.ndim == 1:
        y_pred = y_pred[:, np.newaxis]
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)
    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)
    return 'tran_logloss', np.average(loss)
'''

def train_xgb(X, y, params):
    print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, config.random_seed))
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.1, random_state=config.random_seed)

    xg_train = xgb.DMatrix(X, label=y)
    xg_val = xgb.DMatrix(X_val, label=y_val)

    watchlist  = [(xg_train,'train'), (xg_val,'eval')]
    return xgb.train(params, xg_train, ROUNDS, watchlist, early_stopping_rounds=20)

def predict_xgb(clr, X_test):
    return clr.predict(xgb.DMatrix(X_test))

def main():
    params = {}

    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'logloss'
    params['eta'] = 0.02
    params['max_depth'] = 8
    params['subsample'] = 0.6
    params['silent'] = 1
    params['base_score'] = 0.2
    params['scale_pos_weight'] = 0.360574285

    train_abhishek = pd.read_csv("{}/train_abhishek.csv".format(config.feat_folder))
    test_abhishek = pd.read_csv("{}/test_abhishek.csv".format(config.feat_folder))

    train_fuzz_feat = pd.read_csv("{}/train_fuzz_feat.csv".format(config.feat_folder))
    test_fuzz_feat = pd.read_csv("{}/test_fuzz_feat.csv".format(config.feat_folder))

    train_determiner = pd.read_csv("{}/train_determiner.csv".format(config.feat_folder))
    test_determiner = pd.read_csv("{}/test_determiner.csv".format(config.feat_folder))

    train_hamming = pd.read_csv("{}/train_hamming.csv".format(config.feat_folder))
    test_hamming = pd.read_csv("{}/test_hamming.csv".format(config.feat_folder))

    train_leak_feat = pd.read_csv("{}/train_leak_feat.csv".format(config.feat_folder))
    test_leak_feat = pd.read_csv("{}/test_leak_feat.csv".format(config.feat_folder))

    train_len = pd.read_csv("{}/train_len.csv".format(config.feat_folder))
    test_len = pd.read_csv("{}/test_len.csv".format(config.feat_folder))

    train_max_kcore = pd.read_csv("{}/train_max_kcore.csv".format(config.feat_folder))
    test_max_kcore = pd.read_csv("{}/test_max_kcore.csv".format(config.feat_folder))

    train_ngram = pd.read_csv("{}/train_ngram.csv".format(config.feat_folder))
    test_ngram = pd.read_csv("{}/test_ngram.csv".format(config.feat_folder))

    train_nonascii_jaccard = pd.read_csv("{}/train_nonascii_jaccard.csv".format(config.feat_folder))
    test_nonascii_jaccard = pd.read_csv("{}/test_nonascii_jaccard.csv".format(config.feat_folder))

    train_punctuation = pd.read_csv("{}/train_punctuation.csv".format(config.feat_folder))
    test_punctuation = pd.read_csv("{}/test_punctuation.csv".format(config.feat_folder))

    train_set_distance = pd.read_csv("{}/train_set_distance.csv".format(config.feat_folder))
    test_set_distance = pd.read_csv("{}/test_set_distance.csv".format(config.feat_folder))

    train_stopwords = pd.read_csv("{}/train_stopwords.csv".format(config.feat_folder))
    test_stopwords = pd.read_csv("{}/test_stopwords.csv".format(config.feat_folder))

    train_tfidf = pd.read_csv("{}/train_tfidf.csv".format(config.feat_folder))
    test_tfidf = pd.read_csv("{}/test_tfidf.csv".format(config.feat_folder))

    train_word2vec_feat = pd.read_csv("{}/train_word2vec_feat.csv".format(config.feat_folder))
    test_word2vec_feat = pd.read_csv("{}/test_word2vec_feat.csv".format(config.feat_folder))
    
    train_capital = pd.read_csv("{}/train_capital.csv".format(config.feat_folder))
    test_capital = pd.read_csv("{}/test_capital.csv".format(config.feat_folder))

    train_x = pd.read_csv("{}/train_x.csv".format(config.feat_folder))
    test_x = pd.read_csv("{}/test_x.csv".format(config.feat_folder))

    train_word_stat = pd.read_csv("{}/train_word_stat.csv".format(config.feat_folder))
    test_word_stat = pd.read_csv("{}/test_word_stat.csv".format(config.feat_folder))

    train_wm_feat = pd.read_csv("{}/train_wm_feat.csv".format(config.feat_folder))
    test_wm_feat = pd.read_csv("{}/test_wm_feat.csv".format(config.feat_folder))

    train_pagerank_feats = pd.read_csv("{}/train_pagerank_feats.csv".format(config.feat_folder))
    test_pagerank_feats = pd.read_csv("{}/test_pagerank_feats.csv".format(config.feat_folder))

    train_feat_csv = [train_abhishek,
                      train_fuzz_feat,
                      train_determiner, 
                      train_hamming, 
                      train_leak_feat, 
                      train_len,
                      train_max_kcore,
                      train_ngram,
                      train_nonascii_jaccard,
                      train_punctuation,
                      train_set_distance,
                      train_stopwords,
                      train_tfidf,
                      train_word2vec_feat,
                      train_capital,
                      train_x,
                      train_word_stat,
                      train_wm_feat,
                      train_pagerank_feats]

    test_feat_csv = [test_abhishek,
                     test_fuzz_feat,
                     test_determiner, 
                     test_hamming, 
                     test_leak_feat, 
                     test_len,
                     test_max_kcore,
                     test_ngram,
                     test_nonascii_jaccard,
                     test_punctuation,
                     test_set_distance,
                     test_stopwords,
                     test_tfidf,
                     test_word2vec_feat,
                     test_capital,
                     test_x,
                     test_word_stat,
                     test_wm_feat,
                     test_pagerank_feats]

    x_train = pd.concat(train_feat_csv, axis=1)
    x_test = pd.concat(test_feat_csv, axis=1)


    df_train = pd.read_csv(config.original_train_data_path)
    df_test = pd.read_csv(config.original_test_data_path)
    y_train = df_train['is_duplicate'].values
    print("Features: {}".format(list(x_train.columns.values)))


    print("Training data: X_train: {}, Y_train: {}, X_test: {}".format(x_train.shape, len(y_train), x_test.shape))
    clr = train_xgb(x_train, y_train, params)
    preds = predict_xgb(clr, x_test)

    print("Writing output...")
    sub = pd.DataFrame()
    sub['test_id'] = df_test['test_id']
    sub['is_duplicate'] = preds
    sub.to_csv("xgb_seed{}_n{}.csv".format(config.random_seed, ROUNDS), index=False)




if __name__ == "__main__":
    main()
    print("Done.")