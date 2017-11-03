
import numpy as np
import pandas as pd

from calc_utils import divide
from nlp_utils import stopwords
from param_config import config



def preprocess(df):
    for col in ["processd_question1", "processd_question2"]:
        df["%s_token" % col] = [[word for word in sentence.split()] for sentence in df[col].astype(str)]

df_train = pd.read_csv(config.processed_train_data_path)
df_test = pd.read_csv(config.processed_test_data_path)
preprocess(df_train)
preprocess(df_test)

def max_len(text1, text2):
    if type(text1) == float:
        text1 = ""
    if type(text2) == float:
        text2 = ""
    return max(len(text1), len(text2))

def abs_diff_len(text1, text2):
    if type(text1) == float:
        text1 = ""
    if type(text2) == float:
        text2 = ""
    return abs(len(text1) - len(text2))

def abs_diff_len_normalized(text1, text2):
    if type(text1) == float:
        text1 = ""
    if type(text2) == float:
        text2 = ""
    return divide(abs(len(text1) - len(text2)) * 1.0, max(len(text1), len(text2)))

def char_len(token):
    return len("".join(token))

def char_abs_diff_len(token1, token2):
    return abs(len("".join(token1)) - len("".join(token2)))


def char_abs_diff_len_normalized(len1, len2):
    return divide(abs(len1 - len2) * 1.0, max(len1, len2))

df_train["q1_text_len"] = df_train["processd_question1"].apply(lambda x: len(str(x)))
df_test["q1_text_len"] = df_test["processd_question1"].apply(lambda x: len(str(x)))

df_train["q2_text_len"] = df_train["processd_question2"].apply(lambda x: len(str(x)))
df_test["q2_text_len"] = df_test["processd_question2"].apply(lambda x: len(str(x)))

#df_train["max_text_len"] = df_train.apply(lambda row: max_len(row["processd_question1"], row["processd_question2"]), axis=1)
#df_test["max_text_len"] = df_test.apply(lambda row: max_len(row["processd_question1"], row["processd_question2"]), axis=1)

df_train["text_abs_diff_len"] = df_train.apply(lambda row: abs_diff_len(row["processd_question1"], row["processd_question2"]), axis=1)
df_test["text_abs_diff_len"] = df_test.apply(lambda row: abs_diff_len(row["processd_question1"], row["processd_question2"]), axis=1)

#df_train["text_log_abs_diff_len"] = np.log(df_train["text_abs_diff_len"] + 1)
#df_test["text_log_abs_diff_len"] = np.log(df_test["text_abs_diff_len"] + 1)

df_train["ratio_text_len"] = df_train["q1_text_len"].apply(lambda x: x if x > 0.0 else 1.0) / df_train["q2_text_len"].apply(lambda x: x if x > 0.0 else 1.0)
df_test["ratio_text_len"] = df_test["q1_text_len"].apply(lambda x: x if x > 0.0 else 1.0) / df_test["q2_text_len"].apply(lambda x: x if x > 0.0 else 1.0)

#df_train["text_abs_diff_len_normalized"] = df_train.apply(lambda row: abs_diff_len_normalized(row["processd_question1"], row["processd_question2"]), axis=1)
#df_test["text_abs_diff_len_normalized"]= df_test.apply(lambda row: abs_diff_len_normalized(row["processd_question1"], row["processd_question2"]), axis=1)

df_train["q1_word_len"] = df_train["processd_question1_token"].apply(len)
df_test["q1_word_len"] = df_test["processd_question1_token"].apply(len)

df_train["q2_word_len"] = df_train["processd_question2_token"].apply(len)
df_test["q2_word_len"] = df_test["processd_question2_token"].apply(len)

#df_train["max_word_len"] = df_train.apply(lambda row: max_len(row["processd_question1_token"], row["processd_question2_token"]), axis=1)
#df_test["max_word_len"] = df_test.apply(lambda row: max_len(row["processd_question1_token"], row["processd_question2_token"]), axis=1)

df_train["word_abs_diff_len"] = df_train.apply(lambda row: abs_diff_len(row["processd_question1_token"], row["processd_question2_token"]), axis=1)
df_test["word_abs_diff_len"] = df_test.apply(lambda row: abs_diff_len(row["processd_question1_token"], row["processd_question2_token"]), axis=1)

#df_train["word_abs_diff_len_normalized"] = df_train.apply(lambda row: abs_diff_len_normalized(row["processd_question1_token"], row["processd_question2_token"]), axis=1)
#df_test["word_abs_diff_len_normalized"] = df_test.apply(lambda row: abs_diff_len_normalized(row["processd_question1_token"], row["processd_question2_token"]), axis=1)

df_train["q1_char_len"] = df_train["processd_question1_token"].apply(char_len)
df_test["q1_char_len"] = df_test["processd_question1_token"].apply(char_len)

df_train["q2_char_len"] = df_train["processd_question2_token"].apply(char_len)
df_test["q2_char_len"] = df_test["processd_question2_token"].apply(char_len)

df_train["char_abs_diff_len"] = df_train.apply(lambda row: char_abs_diff_len(row["processd_question1_token"], row["processd_question2_token"]), axis=1)
df_test["char_abs_diff_len"] = df_test.apply(lambda row: char_abs_diff_len(row["processd_question1_token"], row["processd_question2_token"]), axis=1)

df_train["q1_avg_char_len_per_word"] = df_train.apply(lambda row: divide(row["q1_char_len"], row["q1_word_len"]), axis=1)
df_test["q1_avg_char_len_per_word"] = df_test.apply(lambda row: divide(row["q1_char_len"], row["q1_word_len"]), axis=1)

df_train["q2_avg_char_len_per_word"] = df_train.apply(lambda row: divide(row["q2_char_len"], row["q2_word_len"]), axis=1)
df_test["q2_avg_char_len_per_word"] = df_test.apply(lambda row: divide(row["q2_char_len"], row["q2_word_len"]), axis=1)

#df_train["max_avg_char_len_per_word"] = df_train.apply(lambda row: max(row["q1_avg_char_len_per_word"], row["q2_avg_char_len_per_word"]), axis=1)
#df_test["max_avg_char_len_per_word"] = df_test.apply(lambda row: max(row["q1_avg_char_len_per_word"], row["q2_avg_char_len_per_word"]), axis=1)

#df_train["diff_avg_char_len_per_word"] = df_train.apply(lambda row: abs(row["q1_avg_char_len_per_word"] - row["q2_avg_char_len_per_word"]), axis=1)
#df_test["diff_avg_char_len_per_word"] = df_test.apply(lambda row: abs(row["q1_avg_char_len_per_word"] - row["q2_avg_char_len_per_word"]), axis=1)

df_train["diff_avg_char_len_per_word_normalized"] = df_train.apply(lambda row: char_abs_diff_len_normalized(row["q1_avg_char_len_per_word"], row["q2_avg_char_len_per_word"]), axis=1)
df_test["diff_avg_char_len_per_word_normalized"] = df_test.apply(lambda row: char_abs_diff_len_normalized(row["q1_avg_char_len_per_word"], row["q2_avg_char_len_per_word"]), axis=1)



df_train['text_diff_len'] = df_train['q1_text_len'] - df_train['q2_text_len']
df_train['diff_char_len'] = df_train['q1_char_len'] - df_train['q2_char_len']
df_train['diff_avg_char_len_per_word'] = df_train['q1_avg_char_len_per_word'] - df_train['q2_avg_char_len_per_word']

df_test['text_diff_len'] = df_test['q1_text_len'] - df_test['q2_text_len']
df_test['diff_char_len'] = df_test['q1_char_len'] - df_test['q2_char_len']
df_test['diff_avg_char_len_per_word'] = df_test['q1_avg_char_len_per_word'] - df_test['q2_avg_char_len_per_word']

feats = ["q1_text_len", "q2_text_len", "text_diff_len", "ratio_text_len","text_abs_diff_len",
         "q1_word_len", "q2_word_len","word_abs_diff_len",
         "q1_char_len", "q2_char_len","diff_char_len","char_abs_diff_len",
         "q1_avg_char_len_per_word", "q2_avg_char_len_per_word", "diff_avg_char_len_per_word","diff_avg_char_len_per_word_normalized"]

df_train[feats].to_csv("{}/train_len.csv".format(config.feat_folder), index=False)
df_test[feats].to_csv("{}/test_len.csv".format(config.feat_folder), index=False)


