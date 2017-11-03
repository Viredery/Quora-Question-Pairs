import pandas as pd
import numpy as np

from calc_utils import divide
from nlp_utils import stopwords
from param_config import config

def preprocess(df):
    for col in ["processd_question1", "processd_question2"]:
        df["%s_token" % col] = [[word for word in sentence.split()] for sentence in df[col].astype(str)]

def gen_interrogative_word(df, word):
    df["q1_" + word] = df["processd_question1_token"].apply(lambda sentence: (word in sentence) * 1)
    df["q2_" + word] = df["processd_question2_token"].apply(lambda sentence: (word in sentence) * 1)
    df["both_" + word] = df["q1_" + word] * df["q2_" + word]

if __name__ == "__main__":
    df_train = pd.read_csv(config.processed_train_data_path)
    df_test = pd.read_csv(config.processed_test_data_path)
    preprocess(df_train)
    preprocess(df_test)

    gen_interrogative_word(df_train, 'how')
    gen_interrogative_word(df_train, 'what')
    gen_interrogative_word(df_train, 'which')
    gen_interrogative_word(df_train, 'who')
    gen_interrogative_word(df_train, 'where')
    gen_interrogative_word(df_train, 'when')
    gen_interrogative_word(df_train, 'why')

    gen_interrogative_word(df_test, 'how')
    gen_interrogative_word(df_test, 'what')
    gen_interrogative_word(df_test, 'which')
    gen_interrogative_word(df_test, 'who')
    gen_interrogative_word(df_test, 'where')
    gen_interrogative_word(df_test, 'when')
    gen_interrogative_word(df_test, 'why')


    feats = ["q1_how", "q2_how", "both_how", "q1_what", "q2_what", "both_what", "q1_which", "q2_which", "both_which",
             "q1_who", "q2_who", "both_who", "q1_where", "q2_where", "both_where", "q1_when", "q2_when", "both_when",
             "q1_why", "q2_why", "both_why"]

    df_train[feats].to_csv("{}/train_determiner.csv".format(config.feat_folder), index=False)
    df_test[feats].to_csv("{}/test_determiner.csv".format(config.feat_folder), index=False)