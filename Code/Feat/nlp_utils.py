import re
import nltk
from nltk.stem.porter import PorterStemmer
from gensim.models import Word2Vec

from replacer import CsvReplacer, BaseReplacer
from param_config import config

#############
# stopwords #
#############
stopwords = set(nltk.corpus.stopwords.words('english'))

###########
# extract #
###########
def extract_nonascii(sentence):
    nonascii_list = re.findall(r"[^\x00-\x7f]+", sentence)
    return "".join(nonascii_list)

###########
# replace #
###########
def replace(row, replacer):
    names = ["processd_question1", "processd_question2"]
    for name in names:
        row[name] = replacer.replace(row[name])
    return row

nonascii_replacer = CsvReplacer("%s/nonascii_replacement.csv" % config.data_folder)
def replace_nonascii(row):
    return replace(row, nonascii_replacer)

abbreviation_replacer = CsvReplacer("%s/abbreviation_replacement.csv" % config.data_folder)
def replace_abbreviation(row):
    return replace(row, abbreviation_replacer)

synonym_replacer = CsvReplacer("%s/synonyms_replacement.csv" % config.data_folder)
def replace_synonym(row):
    return replace(row, synonym_replacer)

math_mark_deleter = BaseReplacer([("\[math\]", " "), ("\[/math\]", " ")])
def delete_math_mark(row):
    return replace(row, math_mark_deleter)

nonascii_and_punctuation_deleter = BaseReplacer([("[^a-zA-Z0-9]", " ")])
def delete_nonascii_and_punctuation(row):
    return replace(row, nonascii_and_punctuation_deleter)


########
# stem #
########
porter_stemmer = PorterStemmer()
def stem_str(tokens):
    tokens_stemmed = [porter_stemmer.stem(token).strip() for token in tokens if token not in stopwords]
    return tokens_stemmed