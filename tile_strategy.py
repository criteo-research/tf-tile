'''
you can add your own straget for tile-strategy--bucketization/discretization of data. 
here we use uniform
'''

import numpy as np


class TileStrategy(object):

	def __init__(self,var_value_range):

		self.var_value_range = var_value_range

	def uniform(self,num_buckets):
		'''
		all vars get the same number of buckets
		'''
		intervals_per_var = dict()

		for key,v in self.var_value_range.items():
			intervals_per_var[key] = np.linspace(v[0],v[1],num_buckets+1)

		return intervals_per_var


	def custom(num_buckets_per_var):

		'''
		you can define a different number of buckets for each variable and also select the
		intervals non-uniformly through your own policy and data-distribution--for example using
		kernel density estimation methods

		'''
		pass
