
import numpy as np
import pandas as pd
import functools
from calc_utils import divide
from nlp_utils import stopwords
from param_config import config



def preprocess(df):
    for col in ["processd_question1", "processd_question2"]:
        df[col] = [[word for word in sentence.split()] for sentence in df[col].astype(str)]

df_train = pd.read_csv(config.processed_train_data_path)
df_test = pd.read_csv(config.processed_test_data_path)
preprocess(df_train)
preprocess(df_test)

def total_unique_words(row):
    return len(set(str(row['question1'])).union(set(str(row['question2']))))

def total_unq_words_stop(row, stops):
    return len([x for x in set(str(row['question1'])).union(set(str(row['question1']))) if x not in stops])

def wc_ratio(row):
    l1 = len(row['processd_question1'])*1.0 
    l2 = len(row['processd_question2'])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique(row):
    return abs(len(set(row['processd_question1'])) - len(set(row['processd_question2'])))

def wc_ratio_unique(row):
    l1 = len(set(row['processd_question1'])) * 1.0
    l2 = len(set(row['processd_question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def wc_diff_unique_stop(row, stops=None):
    return abs(len([x for x in set(row['processd_question1']) if x not in stops]) - len([x for x in set(row['processd_question2']) if x not in stops]))

def wc_ratio_unique_stop(row, stops=None):
    l1 = len([x for x in set(row['processd_question1']) if x not in stops])*1.0 
    l2 = len([x for x in set(row['processd_question2']) if x not in stops])
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def same_start_word(row):
    if not row['processd_question1'] or not row['processd_question2']:
        return np.nan
    return int(row['processd_question1'][0] == row['processd_question2'][0])

def char_ratio(row):
    l1 = len(''.join(row['processd_question1'])) 
    l2 = len(''.join(row['processd_question2']))
    if l2 == 0:
        return np.nan
    if l1 / l2:
        return l2 / l1
    else:
        return l1 / l2

def char_diff_unique_stop(row, stops=None):
    return abs(len(''.join([x for x in set(row['processd_question1']) if x not in stops])) - len(''.join([x for x in set(row['processd_question2']) if x not in stops])))

data = pd.concat([df_train, df_test])
X = pd.DataFrame()
X['wc_ratio'] = data.apply(wc_ratio, axis=1, raw=True)
X['wc_diff_unique'] = data.apply(wc_diff_unique, axis=1, raw=True)
X['wc_ratio_unique'] = data.apply(wc_ratio_unique, axis=1, raw=True)

f = functools.partial(wc_diff_unique_stop, stops=stopwords)    
X['wc_diff_unq_stop'] = data.apply(f, axis=1, raw=True)
f = functools.partial(wc_ratio_unique_stop, stops=stopwords)    
X['wc_ratio_unique_stop'] = data.apply(f, axis=1, raw=True) 

X['same_start'] = data.apply(same_start_word, axis=1, raw=True) 

f = functools.partial(char_diff_unique_stop, stops=stopwords) 
X['char_diff_unq_stop'] = data.apply(f, axis=1, raw=True) 

X['total_unique_words'] = data.apply(total_unique_words, axis=1, raw=True)

f = functools.partial(total_unq_words_stop, stops=stopwords)
X['total_unq_words_stop'] = data.apply(f, axis=1, raw=True)
X['char_ratio'] = data.apply(char_ratio, axis=1, raw=True)

df_train = X[:df_train.shape[0]]
df_test = X[df_train.shape[0]:]


feats = ["wc_ratio", "wc_diff_unique", "wc_ratio_unique", "wc_diff_unq_stop", "wc_ratio_unique_stop", "same_start",
         "char_diff_unq_stop", "total_unique_words", "total_unq_words_stop", "char_ratio"]
df_train[feats].to_csv("{}/train_word_stat.csv".format(config.feat_folder), index=False)
df_test[feats].to_csv("{}/test_word_stat.csv".format(config.feat_folder), index=False)
