import pandas as pd
import numpy as np

from calc_utils import divide
from ngram import get_bigram, get_trigram
from nlp_utils import stopwords
from param_config import config

df_train = pd.read_csv(config.original_train_data_path)
df_test = pd.read_csv(config.original_test_data_path)

def generate_qid_graph_table(row):

    hash_key1 = row["question1"]
    hash_key2 = row["question2"]

    qid_graph.setdefault(hash_key1, []).append(hash_key2)
    qid_graph.setdefault(hash_key2, []).append(hash_key1)

qid_graph = {}
df_train.apply(generate_qid_graph_table, axis=1)
df_test.apply(generate_qid_graph_table, axis=1)

def pagerank():
    MAX_ITER = 20
    d = 0.85
    pagerank_dict = {i:1/len(qid_graph) for i in qid_graph}
    num_nodes = len(pagerank_dict)
    for iter in range(0, MAX_ITER):
        for node in qid_graph:    
            local_pr = 0
            for neighbor in qid_graph[node]:
                local_pr += pagerank_dict[neighbor]/len(qid_graph[neighbor])
            pagerank_dict[node] = (1-d)/num_nodes + d*local_pr
    return pagerank_dict
pagerank_dict = pagerank()

def get_pagerank_value(row):
    return pd.Series({
        "q1_pr": pagerank_dict[row["question1"]],
        "q2_pr": pagerank_dict[row["question2"]]
    })
train_pagerank_feats = df_train.apply(get_pagerank_value, axis=1)
test_pagerank_feats = df_test.apply(get_pagerank_value, axis=1)

train_pagerank_feats.to_csv("{}/train_pagerank_feats.csv".format(config.feat_folder), index=False)
test_pagerank_feats.to_csv("{}/test_pagerank_feats.csv".format(config.feat_folder), index=False)