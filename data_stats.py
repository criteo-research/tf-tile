import pandas as pd
from collections import defaultdict

FILE_NAME = "winequality-red.csv"

df = pd.read_csv(FILE_NAME,sep=';')



def get_feature_range():
	feature_range = defaultdict(list)
	max_vals = df.max()
	min_vals = df.min()

	for k,v in min_vals.iteritems():
		k = name_to_valid_str(k)

		feature_range[k].append(v)

	for k,v in max_vals.iteritems():
		k = name_to_valid_str(k)
		feature_range[k].append(v)

	del feature_range['quality']
	return feature_range

def name_to_valid_str(input_str):
	return '_'. join(input_str.split(' '))






