def get_unigram(word_list):
    if type(word_list) != list:
        return []
    return word_list

def get_bigram(word_list, join_string='_', skip=0):
    num = len(word_list)
    if num < 2:
        return get_unigram(word_list)
    bigrams = []
    for i in range(num-1):
        for k in range(1, skip+2):
            if i + k < num:
                bigrams.append(join_string.join([word_list[i], word_list[i+k]]))
    return bigrams

def get_trigram(word_list, join_string='_', skip=0):
    num = len(word_list)
    if num < 3:
        return get_bigram(word_list, join_string, skip)
    trigrams = []
    for i in range(num-2):
        for k1 in range(1, skip+2):
            for k2 in range(1, skip+2):
                if i + k1 + k2 < num:
                    trigrams.append(join_string.join([word_list[i], word_list[i+k1], word_list[i+k1+k2]]))
    return trigrams


def get_biterm(word_list, join_string='_'):
    num = len(word_list)
    if num < 2:
        return get_unigram(word_list)
    biterms = []
    for i in range(num-1):
         for j in range(i+1, num):
            biterms.append(join_string.join(word_list[i], word_list[j]))
    return biterms

def get_triterm(word_list, join_string='_'):
    num = len(word_list)
    if num < 3:
        return get_biterm(word_list, join_string)
    triterms = []
    for i in range(num-2):
        for j in range(i+1, num-1):
            for k in range(j+1, num):
                triterms.append(join_string.join(word_list[i], word_list[j], word_list[k]))
    return triterms
