import pandas as pd
import numpy as np


from gen_punctuation_feat import question_mark_num_diff, math_tag_num_diff, height_data_diff
from nlp_utils import replace_nonascii, replace_abbreviation, replace_synonym, delete_math_mark, delete_nonascii_and_punctuation, stem_str
from gen_nonascii_distance import dist_jaccrard_nonascii
from param_config import config

np.random.seed(config.random_seed)

print("Loading data...")

def load_original_data():
    def convert_to_str(sentence):
        sentence["question1"] = "" if type(sentence["question1"]) != str else sentence["question1"]
        sentence["question2"] = "" if type(sentence["question2"]) != str else sentence["question2"]
        return sentence
    df_train = pd.read_csv(config.original_train_data_path).fillna("")
    df_test = pd.read_csv(config.original_test_data_path).fillna("")
    df_train = df_train.apply(convert_to_str, axis=1)
    df_test = df_test.apply(convert_to_str, axis=1)
    num_train, num_test = df_train.shape[0], df_test.shape[0]
    return df_train, df_test, num_train, num_test

df_train, df_test, num_train, num_test = load_original_data()

print("Done.")

print("Preprocessing data 1: lower...")

for name in ["question1", "question2"]:
    df_train["processd_%s" % name] = df_train[name].str.lower()
    df_test["processd_%s" % name] = df_test[name].str.lower()

print("Done.")


print("Preprocessing data 2: handle nonascii characters...")

df_train = df_train.apply(replace_nonascii, axis=1)
df_test = df_test.apply(replace_nonascii, axis=1)

# generate nonascii_jaccard feature
df_train["nonascii_jaccard"] = df_train.apply(dist_jaccrard_nonascii, axis=1)
df_test["nonascii_jaccard"] = df_test.apply(dist_jaccrard_nonascii, axis=1)

df_train["nonascii_jaccard"].to_csv("{}/train_nonascii_jaccard.csv".format(config.feat_folder), index=False)
df_test["nonascii_jaccard"].to_csv("{}/test_nonascii_jaccard.csv".format(config.feat_folder), index=False)

print("Done.")


print("Preprocessing data 3: replace words and punctuations...")

df_train = df_train.apply(replace_abbreviation, axis=1)
df_test = df_test.apply(replace_abbreviation, axis=1)
df_train = df_train.apply(replace_synonym, axis=1)
df_test = df_test.apply(replace_synonym, axis=1)


print("Done.")


print("Preprocessing data 4: extract the information of punctuations...")

df_train["question_mark_num_diff"] = df_train.apply(question_mark_num_diff, axis=1)
df_test["question_mark_num_diff"] = df_test.apply(question_mark_num_diff, axis=1)
df_train["math_tag_num_diff"] = df_train.apply(math_tag_num_diff, axis=1)
df_test["math_tag_num_diff"] = df_test.apply(math_tag_num_diff, axis=1)
df_train["height_data_diff"] = df_train.apply(height_data_diff, axis=1)
df_test["height_data_diff"] = df_test.apply(height_data_diff, axis=1)

feats = ["question_mark_num_diff", "math_tag_num_diff", "height_data_diff"]
df_train[feats].to_csv("{}/train_punctuation.csv".format(config.feat_folder), index=False)
df_test[feats].to_csv("{}/test_punctuation.csv".format(config.feat_folder), index=False)

df_train = df_train.apply(delete_math_mark, axis=1)
df_test = df_test.apply(delete_math_mark, axis=1)

print("Done.")

print("Preprocessing data 5: delete all nonascii characters and punctuations...")

df_train = df_train.apply(delete_nonascii_and_punctuation, axis=1)
df_test = df_test.apply(delete_nonascii_and_punctuation, axis=1)

print("Preprocessing data 6: stemming...")

df_train = df_train.apply(stem_str, axis=1)
df_test = df_test.apply(stem_str, axis=1)

print("Done.")

df_train[["question1", "question2", "processd_question1", "processd_question2"]].to_csv(config.processed_train_data_path)
df_test[["question1", "question2", "processd_question1", "processd_question2"]].to_csv(config.processed_test_data_path)