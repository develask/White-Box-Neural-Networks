import numpy as np
import wbnn

# This is an example of defining a new layer in WBNN.
# It is a lookup table that given an index gives a word vector of the corresponding word.
# On the other hand note that this layer has no parameters and therefore is not trainable.
# Also, it is not differentiable but it is okay since it will only be used as the first layer just
# after the input layer. 

class LookUpTable(wbnn.layers.Layer):
	def __init__(self, lookup_matrix, name):
		super(LookUpTable, self).__init__(name)
		self.lookup = lookup_matrix
	def initialize(self):
		assert len(self.prev) == 1 and len(self.prev_recurrent) == 0
	def get_Output_dim(self):
		return self.lookup.shape[1]
	def prop(self):
		prev_out = self.prev[0].get_Output()
		out = self.lookup[prev_out, :].reshape((prev_out.shape[0], self.get_Output_dim()))
		self.a = self.a + [out]
		return out
	def update(self):
		pass
	def copy(self):
		return LookUpTable(self.lookup, self.name)
	def __save__dict__(self):
		return {'lookup': self.lookup}, [None, self.name]
	def __load__dict__(self, d):
		self.lookup = d['lookup']

wbnn.layers.LookUpTable = LookUpTable
