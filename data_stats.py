import pandas as pd
from collections import defaultdict
import numpy as np
import csv
import winequality
FILE_NAME = "winequality-red.csv"

df = pd.read_csv(FILE_NAME,sep=';')

def outliers(df, threshold, columns):
    for col in columns: 
        mask = df[col] > float(threshold)*df[col].std()+df[col].mean()
        df.loc[mask == True,col] = np.nan
        mean_property = df.loc[:,col].mean()
        df.loc[mask == True,col] = mean_property
    return df
#here we bin the data based on three class: Bad, average and Good
column_list = df.columns.tolist()
threshold = 5
df_cleaned = df.copy()
df_cleaned = outliers(df_cleaned, threshold, column_list[0:-1])

bins = [3, 5, 6, 8]

df_cleaned['category'] = pd.cut(df_cleaned.quality, bins, labels=['Bad', 'Average', 'Good'])

df_binary_cat = df_cleaned[df_cleaned['category'].isin(['Bad', 'Good'])].copy()

df=df_binary_cat.replace('Good',1).replace('Bad',0).copy() 

# del df['quality']
# df.to_csv('df_to_csv.csv',sep=';',index=False)

# writeFile = open('newtest.csv ','w')

# writer = csv.writer(writeFile,delimiter=';')
# header = [[k] for k in winequality.FEATURES+['category']]
# writer.writerows(header)
# for index, row in df.iterrows():
# 	list_row = row.to_list()
# 	new_row = [[k] for k in list_row[0:11]+[int(list_row[-1])]] 
# 	writer.writerows(new_row) 

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
	del feature_range['category']
	return feature_range

def name_to_valid_str(input_str):
	return '_'. join(input_str.split(' '))






