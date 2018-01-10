import numpy as np
import random
import math
import time
import json
import layers
import os

from layers import Input

class Loss():
	def __init__(self, name):
		self.name = name

	def ff(self, a, y):
		if self.name == "mse":
			return sum((y-a)**2)/2
		elif self.name == "ce1":
			return sum(-y*np.log(a+0.0001)-(1-y)*np.log(1.0001-a))

	def grad(self, a, y):
		# returns the gradient with respect to all the components of a.
		# Thus the dimension of the output np.array is the same that a, i.e. [output_dim, 1]
		if self.name == "mse":
			return a - y
		elif self.name == "ce1":
			if np.sum(a == 0) == 0 and np.sum(a==1) == 0:
				if np.isnan(-y/(a) + (1-y)/(1-a)).any():
					pass

				return -y/(a) + (1-y)/(1-a)
			else:
				return -(y)/(a+0.0001) + (1-y)/(1.0001-a)
	

class DNN():
	def __init__(self, name):
		self.name = name
		self.inputs = []
		self.next = []
		self.prev = []
		self.next_rercurrent = []
		self.prev_recurrent = []
		self.output_layer = None
		self.loss = Loss("ce1")

	def calculate_layer_order(self):
		"""
		Since DNN object contains a graph representation,
		this function calculates the propagation order.
		Using the reverse order, work for backpropagation.
		"""

		def importing_dnn(nn):
			# modify conections on next and prev
			for inp in nn.inputs:
				l_prev = inp.prev
				try:
					idx = l_prev.next.index(nn)
					l_prev.next[idx:idx+1] = inp.next
					for l_hidden in inp.next:
						l_hidden.prev[l_hidden.prev.index(inp)] = l_prev
				except ValueError:
					idx = l_prev.next_rercurrent.index(nn)
					assert idx != -1
					l_prev.next_rercurrent[idx:idx+1] = inp.next
					for l_hidden in inp.next:
						l_hidden.prev_recurrent[l_hidden.prev_recurrent.index(inp)] = l_prev
			nn.prop_order[-1].next_rercurrent += nn.next_rercurrent
			for l in nn.next_rercurrent:
				l.prev_recurrent[l.prev_recurrent.index(nn)] = nn.prop_order[-1]
			nn.prop_order[-1].next += nn.next
			for l in nn.next:
				l.prev[l.prev.index(nn)] = nn.prop_order[-1]

		self.prop_order = self.inputs.copy()
		i = 0
		while i < len(self.prop_order):
			layer = self.prop_order[i]
			for l in layer.next:
				if sum(map(lambda l_: not l_ in self.prop_order and not l_ is l, l.prev)) == 0 and not l in self.prop_order:
					to_add = [l]
					if isinstance(l, DNN):
						l.calculate_layer_order()
						importing_dnn(l)
						to_add = l.prop_order[len(l.inputs):]

					self.prop_order += to_add
			i+=1

	def initialize(self, init_layers = True):
		"""Initialize each layer, and generate the layer order."""
		if not hasattr(self, 'prop_order'):
			self.calculate_layer_order()
		for layer in self.prop_order:
			if init_layers:
				layer.initialize()
			if len(layer.next) == 0:
				self.output_layer = layer

		found = False
		self.last_rec_idx = 0
		for i in range(-1,-len(self.prop_order),-1):
			if (len(self.prop_order[i].next_rercurrent)>0 or self.prop_order[i].get_in_recurrent_part())and not found:
				self.last_rec_idx = i
				found = True
			self.prop_order[i].set_in_recurrent_part(found)

	def add_inputs(self, *layers):
		for layer in layers:
			self.inputs.append(layer)

	def addNext(self, layer):
		self.next.append(layer)
		layer.__addPrev__(self)

	def __addPrev__(self, layer):
		idx = len(self.prev_recurrent)+len(self.prev)
		assert idx < len(self.inputs)
		assert layer.get_Output_dim() == self.inputs[idx].get_Output_dim()
		self.prev.append(layer)
		self.inputs[idx].prev = layer

	def addNextRecurrent(self, layer):
		self.next_rercurrent.append(layer)
		layer.__addPrevRecurrent__(self)

	def __addPrevRecurrent__(self, layer):
		idx = len(self.prev_recurrent)+len(self.prev)
		assert idx < len(self.inputs)
		assert layer.get_Output_dim() == self.inputs[idx].get_Output_dim()
		self.prev_recurrent.append(layer)
		self.inputs[idx].prev = layer

	def get_Output(self, t=0):
		return self.output_layer.get_Output(t=t)

	def prop(self, inp):
		for layer in self.prop_order:
			layer.reset(inp[0][0].shape[0])
		out = []
		for inp_t in inp:
			inp_t_cop = inp_t.copy()
			out.append([layer.prop(inp_t_cop.pop(0)) if isinstance(layer, Input) else layer.prop() for layer in self.prop_order][-1])
		return out

	def backprop(self, inp, desired_output):
		out = self.prop(inp.copy())[-1]

		self.prop_order[-1].set_loss_error(self.loss.grad(self.prop_order[-1].get_Output(t=0), desired_output))

		for layer in self.prop_order[-2:len(self.inputs)-1:-1]:
			layer.backprop_error(t=0)

		for t in range(-1,-len(inp),-1):
			# We don't have to compute the errors in the output layers until a Recurren layer appears
			for layer in self.prop_order[self.last_rec_idx:len(self.inputs)-1:-1]:
				layer.backprop_error(t=t)

		for layer in self.prop_order[:len(self.inputs)-1:-1]:
			layer.compute_gradients()
		return out

	def reset_gradients(self):
		for k in range(len(self.inputs),len(self.prop_order)):
			self.prop_order[k].reset_grads()

	def apply_to_gradients(self, func):
		for k in range(len(self.inputs),len(self.prop_order)):
			self.prop_order[k].apply_to_gradients(func)

	def get_minibach_grad(self, mini_batch):
		loss = 0
		self.reset_gradients()
		out = self.backprop(*mini_batch)
		# print(j,":", out, mini_batch[j][1])
		loss += np.sum(self.loss.ff(out,mini_batch[1]))

		# Normalizar los gradientes
		self.apply_to_gradients(lambda grad: grad/mini_batch[1].shape[0])
		return loss

	def update_model(self):
		self.apply_to_gradients(lambda x: x*self.lr)
		for k in range(len(self.inputs),len(self.prop_order)):
			self.prop_order[k].update()

	def get_loss_of_data(self, data):
		loss = 0
		len_tr = data[1].shape[0]
		out = self.prop(data[0].copy())
		loss += np.sum(self.loss.ff(out[-1],data[1]))
		return loss/len_tr

	def SGD(self, training_data, batch_size, nb_epochs, lr_start, lr_end,
			func = lambda *x: print(x[0],"training loss:", x[2])):
		self.lr = lr_start
		dec_rate = 1
		if nb_epochs != 1:
			dec_rate = (lr_end/lr_start)**(1/(nb_epochs-1))

		nb_training_examples = training_data[1].shape[0]
		indexes = np.arange(nb_training_examples)

		for j in range(nb_epochs):
			loss = 0

			random.shuffle(indexes)
			for i in range(0,nb_training_examples, batch_size):
				m_b = ([[input_[indexes[i:i+batch_size],:] for input_ in time_step] for time_step in training_data[0]], training_data[1][indexes[i:i+batch_size],:])
				loss_m_b = self.get_minibach_grad(m_b)
				loss += loss_m_b
				self.update_model()

			if func is not None:
				func(j, self, loss/nb_training_examples)
			self.lr *= dec_rate

	def save(self, name = None):
		dir_ = time.strftime(name if name is not None else self.name + "_%Y-%m-%d_%H-%M")
		os.makedirs(dir_)
		inf = {
			'prop_order': [id(l) for l in self.prop_order]
		}
		for layer in self.prop_order:
			l_id = id(layer)
			inf[str(l_id)] = {
				'next': [id(l) for l in layer.next],
				'prev': [id(l) for l in layer.prev],
				'next_rercurrent': [id(l) for l in layer.next_rercurrent],
				'prev_recurrent': [id(l) for l in layer.prev_recurrent],
			}
			layer.save(dir_)
		file_ = json.dumps(inf)
		with open(dir_+"/setup.json", "w") as f:
			f.write(file_)

	def load(dir_):
		with open(dir_+"/setup.json", "r") as f:
			inf = json.loads(f.read())
			layers_ = [layers.Layer.load(dir_+'/'+str(id_)) for id_ in inf['prop_order']]
			layers_id = [str(x) for x in inf['prop_order']]
			del inf['prop_order']
			for l in inf:
				idx = layers_id.index(str(l))
				layers_[idx].next = [layers_[layers_id.index(str(ln))] for ln in inf[l]['next']]
				layers_[idx].prev = [layers_[layers_id.index(str(ln))] for ln in inf[l]['prev']]
				layers_[idx].next_rercurrent = [layers_[layers_id.index(str(ln))] for ln in inf[l]['next_rercurrent']]
				layers_[idx].prev_recurrent = [layers_[layers_id.index(str(ln))] for ln in inf[l]['prev_recurrent']]

			inps = list(filter(lambda x: isinstance(x, Input), layers_))
			nn = DNN(dir_)
			for inp in inps:
				nn.add_inputs(inp)

			nn.initialize(init_layers = False)
			return nn

if __name__ == '__main__':
	import random
	import math
	import time

	from layers import Fully_Connected, LSTM, Softmax
	
	nn = DNN("minibatching")

	x = Input(2, "x")
	h = LSTM(20, "kk")
	out = Softmax(2, "mitiko 2")

	x.addNext(h)
	h.addNext(out)

	nn.add_inputs(x)

	nn.initialize()

	def f(inps):
		x = inps[:,0]
		y = inps[:,1]
		z = inps[:,2]
		v = inps[:,3]
		a = (np.cos(x-2*y+z*v)**2)[:,np.newaxis]
		return np.concatenate((a, 1-a), axis=1)
		# return np.concatenate((a/4, 3*a/4, (1-a)/4, 3*(1-a)/4), axis=1)

	def generate_examples(nb_examples):
		inps = np.random.rand(nb_examples,4)
		outs = f(inps)
		return [[inps[:,:2]],[inps[:,2:]]], outs

	examples_train = generate_examples(10000)

	nn.SGD(examples_train, 128, 10, 0.5, 0.1)

	examples_test = generate_examples(10)
	y = nn.prop(examples_test[0])[-1]
	for i in range(10):
		print(examples_test[0][0][0][i,:], examples_test[0][1][0][i,:], "\n\t--> R", examples_test[1][i,:], "\n\t    P", y[i,:])










