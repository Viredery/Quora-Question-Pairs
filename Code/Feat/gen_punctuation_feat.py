import re

def question_mark_num_diff(row):
    num_question_mark1 = len(re.findall("\?", row["processd_question1"]))
    num_question_mark2 = len(re.findall("\?", row["processd_question2"]))
    return abs(num_question_mark1 - num_question_mark2)

def math_tag_existence(row):
    math_tag1 = len(re.findall("\[math\]", row["processd_question1"])) == 0
    math_tag2 = len(re.findall("\[math\]", row["processd_question2"])) == 0
    if math_tag1 == True and math_tag2 == True:
        return 0
    return 1

def math_tag_num_diff(row):
    num_math_tag1 = len(re.findall("\[math\]", row["processd_question1"]))
    num_math_tag2 = len(re.findall("\[math\]", row["processd_question2"]))
    return abs(num_math_tag1 - num_math_tag2)

def exclamation_mark_diff(row):
    num_exclamation_mark1 = len(re.findall("!", row["processd_question1"]))
    num_exclamation_mark2 = len(re.findall("!", row["processd_question2"]))
    return abs(num_exclamation_mark1 - num_exclamation_mark2)

def full_stop_diff(row):
    num_full_stop1 = len(re.findall("\.", row["processd_question1"]))
    num_full_stop2 = len(re.findall("\.", row["processd_question2"]))
    return abs(num_full_stop1 - num_full_stop2)

def comma_diff(row):
    num_comma1 = len(re.findall(",", row["processd_question1"]))
    num_comma2 = len(re.findall(",", row["processd_question2"]))
    return abs(num_comma1 - num_comma2)

def height_data_diff(row):
    height_data1 = len(re.findall(r"\d+\'\d+\"?", row["processd_question1"])) == 0
    height_data2 = len(re.findall(r"\d+\'\d+\"?", row["processd_question2"])) == 0
    if height_data1 == True and height_data2 == True:
        return 0
    if height_data1 == False and height_data2 == False:
        return 0
    return 1
