import numpy as np
import pandas as pd
from param_config import config

df_train = pd.read_csv(config.original_train_data_path)
df_test = pd.read_csv(config.original_test_data_path)

df_train["q1_caps_count"] = df_train["question1"].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
df_test["q1_caps_count"] = df_test["question1"].apply(lambda x:sum(1 for i in str(x) if i.isupper()))

df_train["q2_caps_count"] = df_train["question2"].apply(lambda x:sum(1 for i in str(x) if i.isupper()))
df_test["q2_caps_count"] = df_test["question2"].apply(lambda x:sum(1 for i in str(x) if i.isupper()))

df_train["diff_caps"] = df_train["q1_caps_count"] - df_train["q2_caps_count"]
df_test["diff_caps"] = df_test["q1_caps_count"] - df_test["q2_caps_count"]

feats = ["q1_caps_count", "q2_caps_count", "diff_caps"]

df_train[feats].to_csv("{}/train_capital.csv".format(config.feat_folder), index=False)
df_test[feats].to_csv("{}/test_capital.csv".format(config.feat_folder), index=False)