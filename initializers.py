import numpy as np

class Initializer():
	def get(self, shape):
		raise NotImplementedError( "Should have implemented this" )

class Constant(Initializer):
	def __init__(self, constant):
		self.constant = constant

	def get(self, shape):
		return np.full(shape, self.constant)


class RandomNormal(Initializer):
	def __init__(self, mean=0.0, stddev=None):
		self.mean = mean
		self.stddev = stddev

	def get(self, shape):
		stddev = self.stddev
		if stddev == None:
			stddev = 1.0 / np.sqrt(np.multiply.reduce(shape))
		return np.random.normal(loc=self.mean, scale=stddev, size=shape)

class RandomUniform(Initializer):
	def __init__(self, minVal=-0.01, maxVal=0.01):
		self.minVal = minVal
		self.maxVal = maxVal
	
	def get(self, shape):
		return np.random.uniform(low=self.minVal, high=self.maxVal, size=shape)
		
		
