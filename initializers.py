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
	def __init__(self, mean=0.0, stddev=0.05):
		self.mean = mean
		self.stddev = stddev

	def get(self, shape):
		return np.random.normal(loc=self.mean, scale=self.stddev, size=shape)

class RandomUniform(Initializer):
	def __init__(self, minVal=0.01, maxVal=0.01):
		self.minVal = minVal
		self.maxVal = maxVal
	
	def get(self, shape):
		return np.random.uniform(low=self.minVal, high=self.maxVal, size=shape)
		
		