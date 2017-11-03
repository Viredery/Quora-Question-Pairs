import numpy as np
import pandas as pd
from collections import Counter

from nlp_utils import stopwords
from param_config import config


def preprocess(df):
    for col in ["processd_question1", "processd_question2"]:
        df[col] = [[word for word in sentence.split()] for sentence in df[col].astype(str)]
df_train = pd.read_csv(config.processed_train_data_path)
df_test = pd.read_csv(config.processed_test_data_path)
preprocess(df_train)
preprocess(df_test)

token_questions = pd.Series(df_train['processd_question1'].tolist() + df_train['processd_question2'].tolist() + df_test['processd_question1'].tolist() + df_test['processd_question2'].tolist()).astype(str)

def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)

eps = 5000 
words = []
for token_question in token_questions:
    words.extend(token_question)
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

def tfidf_word_match_share(row):
    q1words = {}
    q2words = {}
    for word in row["processd_question1"]:
        q1words[word] = 1
    for word in row["processd_question2"]:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0.0
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    q1_weights, q2_weights = [weights.get(w, 0) for w in q1words], [weights.get(w, 0) for w in q2words]
    total_weights = q1_weights + q2_weights
    return np.sum(shared_weights) / np.sum(total_weights)



def tfidf_word_cosine(row):
    q1words = {}
    q2words = {}
    for word in row["processd_question1"]:
        q1words[word] = 1
    for word in row["processd_question2"]:
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        return 0.0
    shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
    q1_weights, q2_weights = [weights.get(w, 0) for w in q1words], [weights.get(w, 0) for w in q2words]
    cosine_denominator = (np.sqrt(np.dot(q1_weights, q1_weights)) * np.sqrt(np.dot(q2_weights, q2_weights)))
    return np.dot(shared_weights, shared_weights) / cosine_denominator

train_tfidf, test_tfidf = pd.DataFrame(), pd.DataFrame()


train_tfidf["tfidf_word_match"] = df_train.astype(str).apply(tfidf_word_match_share, axis=1)
test_tfidf["tfidf_word_match"] = df_test.astype(str).apply(tfidf_word_match_share, axis=1)

train_tfidf["tfidf_cosine"] = df_train.astype(str).apply(tfidf_word_cosine, axis=1)
test_tfidf["tfidf_cosine"] = df_test.astype(str).apply(tfidf_word_cosine, axis=1)



def preprocess_with_stopwords(df):
    for col in ["processd_question1", "processd_question2"]:
        df[col] = [[word for word in sentence.split() if word not in stopwords] for sentence in df[col].astype(str)]
df_train = pd.read_csv(config.processed_train_data_path)
df_test = pd.read_csv(config.processed_test_data_path)
preprocess_with_stopwords(df_train)
preprocess_with_stopwords(df_test)

train_tfidf["tfidf_word_match_without_stops"] = df_train.astype(str).apply(tfidf_word_match_share, axis=1)
test_tfidf["tfidf_word_match_without_stops"] = df_test.astype(str).apply(tfidf_word_match_share, axis=1)

train_tfidf["tfidf_cosine_without_stops"] = df_train.astype(str).apply(tfidf_word_cosine, axis=1)
test_tfidf["tfidf_cosine_without_stops"] = df_test.astype(str).apply(tfidf_word_cosine, axis=1)



feats = ["tfidf_word_match", "tfidf_cosine", "tfidf_word_match_without_stops", "tfidf_cosine_without_stops"]

train_tfidf[feats].to_csv("{}/train_tfidf.csv".format(config.feat_folder), index=False)
test_tfidf[feats].to_csv("{}/test_tfidf.csv".format(config.feat_folder), index=False)

