import wbnn
import os, time
import numpy as np

from functools import reduce

class Logger():

	def __init__(self, folder):
		self.save_folder = folder
		if not os.path.exists(self.save_folder):
			os.makedirs(self.save_folder)

	def write(self, var_name, val_str):
		with open(os.path.join(self.save_folder, var_name+'.dat'), 'a') as f:
			f.write("%f\t%s\n"%(time.time(), val_str))

	def scalar(self, var_name, value):
		self.write('scalar_'+var_name, str(value))

	def histogram(self, var_name, values, intervals=25):
		min_ = np.min(values)
		max_ = np.max(values)
		pad = (max_-min_)/100
		if pad == 0:
			pad = 0.000001
		min_ -= pad
		max_ += pad
		ys = [0]*intervals
		step = (max_-min_)/(intervals)
		xs = np.arange(min_, max_+step, step).tolist()[:intervals+1]
		for i in range(len(xs)-1):
			ys[i] += np.sum( (values>xs[i]) * (values<=xs[i+1]) )
		self.write('hist_'+var_name, str(step)+'\t'+str(list(zip(xs[:-1],ys))))

	def image(self, var_name, values):
		name = 'image_'+var_name
		times = np.asarray([[time.time()]])
		values = values[np.newaxis,...]
		try:
			data = np.load(os.path.join(self.save_folder, name+".npz"))
			times = np.concatenate([data['times'], times], axis = 0)
			values = np.concatenate([data['values'], values], axis = 0)
		except Exception as e:
			pass
		finally:
			np.savez(os.path.join(self.save_folder, name+".npz"), times=times, values=values)
			# np.save(os.path.join(self.save_folder, name+".npy"), values)
