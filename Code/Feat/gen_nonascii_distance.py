import distance

from nlp_utils import extract_nonascii

def dist_jaccard(str1, str2):
    str_set1, str_set2 = set(str1), set(str2)
    if len(str_set1) == 0 and len(str_set2) == 0:
    	return 0
    elif len(str_set1) == 0 or len(str_set2) == 0:
    	return 1
    return distance.jaccard(str_set1, str_set2)

def dist_jaccrard_nonascii(row):
    str1 = extract_nonascii(row["processd_question1"])
    str2 = extract_nonascii(row["processd_question2"])
    return dist_jaccard(str1, str2)