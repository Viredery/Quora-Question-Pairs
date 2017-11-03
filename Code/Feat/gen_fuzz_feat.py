import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz

from param_config import config

if __name__ == "__main__":
    df_train = pd.read_csv(config.processed_train_data_path)
    df_test = pd.read_csv(config.processed_test_data_path)

    df_train['fuzz_qratio'] = df_train.apply(lambda x: fuzz.QRatio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_train['fuzz_WRatio'] = df_train.apply(lambda x: fuzz.WRatio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_train['fuzz_partial_ratio'] = df_train.apply(lambda x: fuzz.partial_ratio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_train['fuzz_partial_token_set_ratio'] = df_train.apply(lambda x: fuzz.partial_token_set_ratio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_train['fuzz_partial_token_sort_ratio'] = df_train.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_train['fuzz_token_set_ratio'] = df_train.apply(lambda x: fuzz.token_set_ratio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_train['fuzz_token_sort_ratio'] = df_train.apply(lambda x: fuzz.token_sort_ratio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)


    df_test['fuzz_qratio'] = df_test.apply(lambda x: fuzz.QRatio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_test['fuzz_WRatio'] = df_test.apply(lambda x: fuzz.WRatio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_test['fuzz_partial_ratio'] = df_test.apply(lambda x: fuzz.partial_ratio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_test['fuzz_partial_token_set_ratio'] = df_test.apply(lambda x: fuzz.partial_token_set_ratio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_test['fuzz_partial_token_sort_ratio'] = df_test.apply(lambda x: fuzz.partial_token_sort_ratio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_test['fuzz_token_set_ratio'] = df_test.apply(lambda x: fuzz.token_set_ratio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)
    df_test['fuzz_token_sort_ratio'] = df_test.apply(lambda x: fuzz.token_sort_ratio(str(x['processd_question1']), str(x['processd_question2'])), axis=1)


    feats = ["fuzz_qratio", "fuzz_WRatio", "fuzz_partial_ratio", "fuzz_partial_token_set_ratio", "fuzz_partial_token_sort_ratio", "fuzz_token_set_ratio", "fuzz_token_sort_ratio"]

    df_train[feats].to_csv("{}/train_fuzz_feat.csv".format(config.feat_folder), index=False)
    df_test[feats].to_csv("{}/test_fuzz_feat.csv".format(config.feat_folder), index=False)