import re
import csv

class BaseReplacer(object):
	def __init__(self, pattern_replacement_pair_list):
		self.pattern_replacement_pair_list = pattern_replacement_pair_list
	def replace(self, text):
		for pattern, replacement in self.pattern_replacement_pair_list:
			text = re.sub(pattern, replacement, text)
		return re.sub("\s+", " ", text).strip()

class CsvReplacer(BaseReplacer):
	def __init__(self, filename):
		self.filename = filename
		self.pattern_replacement_pair_list = []
		for line in csv.reader(open(filename)):
			self.pattern_replacement_pair_list.append((line[0], line[1]))