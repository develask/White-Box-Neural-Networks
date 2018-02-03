import numpy as np
import random
import math
import time
import json
import layers
import os

from layers import Input, Loss

class DNN():
	def __init__(self, name):
		self.name = name
		self.inputs = []
		self.next = []
		self.prev = []
		self.next_rercurrent = []
		self.prev_recurrent = []
		self.output_layers = []

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
			if isinstance(layer, Loss):
				self.output_layers.append(layer)


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
		return [l.get_Output(t=t) for l in self.output_layers]

	def prop(self, inp, desired_output=None):
		for layer in self.prop_order:
			layer.reset(inp[0][0].shape[0])
		out = []
		out_t_cop = None
		for t in range(len(inp)):
			inp_t_cop = inp[t].copy()
			if desired_output and t - (len(inp)-len(desired_output)) >= 0:
				out_t_cop = desired_output[t - (len(inp)-len(desired_output))].copy()
			for layer in self.prop_order:
				if isinstance(layer, Input) or (isinstance(layer, Loss) and out_t_cop):
					if isinstance(layer, Input):
						layer.prop(inp_t_cop.pop(0))
					else:
						layer.prop(out_t_cop.pop(0))
				else:
					layer.prop()

			out.append(self.get_Output())
		return out

	def backprop(self, inp, desired_output):
		out = self.prop(inp.copy(), desired_output)

		for t in range(0,-len(inp),-1):
			for layer in self.prop_order[-1:len(self.inputs)-1:-1]:
				if isinstance(layer, layers.Loss):
					if t <= -len(desired_output):
						layer.backprop_error(t=t, desired_output=None)
					else:
						layer.backprop_error(t=t, desired_output=desired_output[t-1][self.output_layers.index(layer)])
				else:
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
		self.reset_gradients()
		out = self.backprop(*mini_batch)
		loss = sum([sum([ np.sum(out_o) for out_o in out_t]) for out_t in out[len(mini_batch[0])-len(mini_batch[1]):]])

		# Normalizar los gradientes
		self.apply_to_gradients(lambda grad: grad/mini_batch[1][0][0].shape[0])
		return loss

	def update_model(self):
		self.apply_to_gradients(lambda x: x*self.lr)
		for k in range(len(self.inputs),len(self.prop_order)):
			self.prop_order[k].update()

	def get_loss_of_data(self, data):
		len_tr = data[1][0][0].shape[0]
		out = self.prop(data[0].copy(), data[1].copy())
		loss = sum([sum([ np.sum(out_o) for out_o in out_t]) for out_t in out[len(data[0])-len(data[1]):]])
		return loss/len_tr

	def train_step(self, m_b):
		loss_m_b = self.get_minibach_grad(m_b)
		self.update_model()
		return loss_m_b

	def SGD(self, training_data, batch_size, nb_epochs, lr_start, lr_end,
			func = lambda *x: print(x[0],"training loss:", x[2])):
		self.lr = lr_start
		dec_rate = 1
		if nb_epochs != 1:
			dec_rate = (lr_end/lr_start)**(1/(nb_epochs-1))

		nb_training_examples = training_data[1][0][0].shape[0]
		indexes = np.arange(nb_training_examples)

		for j in range(nb_epochs):
			loss = 0

			random.shuffle(indexes)
			for i in range(0,nb_training_examples, batch_size):
				m_b = ([[input_[indexes[i:i+batch_size],:] for input_ in time_step] for time_step in training_data[0]], 
					[[output_[indexes[i:i+batch_size],:] for output_ in time_step] for time_step in training_data[1]])
				loss += self.train_step(m_b)

			if func is not None:
				func(j, self, loss/nb_training_examples)
			self.lr *= dec_rate

	def save(self, name = None):
		dir_ = time.strftime(name if name is not None else self.name + "_%Y-%m-%d_%H-%M")
		try:
			os.makedirs(dir_)
		except Exception as e:
			dir_ = dir_+time.strftime("_%Y-%m-%d_%H-%M")
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

	from layers import Fully_Connected, LSTM, Activation, Loss
	
	nn = DNN("minibatching")

	x = Input(4, "x")
	h1 = Fully_Connected(10, "h1")
	a1 = Activation("sigmoid", "a1")

	h2 = Fully_Connected(10, "h2")
	a2 = Activation("sigmoid", "a2")

	h3 = Fully_Connected(10, "h3")
	a3 = Activation("sigmoid", "a3")

	sm4 = Fully_Connected(2,"sm1")
	a4 = Activation("softmax", "a4")
	sm5 = Fully_Connected(1, "sigmoid_out")
	a5 = Activation("sigmoid", "a5")

	loss1 = Loss("ce1", "loss1")
	loss2 = Loss("ce1", "loss2")

	x.addNext(h1)
	h1.addNext(a1)

	a1.addNext(h2)
	h2.addNext(a2)

	a1.addNext(h3)
	h3.addNext(a3)

	a2.addNext(sm4)
	sm4.addNext(a4)
	a4.addNext(loss1)

	a3.addNext(sm5)
	sm5.addNext(a5)
	a5.addNext(loss2)



	nn.add_inputs(x)

	nn.initialize()

	def f1(inps):
		x = inps[:,0]
		y = inps[:,1]
		z = inps[:,2]
		v = inps[:,3]
		a = (np.cos(x-2*y+z*v)**2)[:,np.newaxis]
		return np.concatenate((a, 1-a), axis=1)
		# return np.concatenate((a/4, 3*a/4, (1-a)/4, 3*(1-a)/4), axis=1)

	def f2(inps):
		x = inps[:,0]
		y = inps[:,1]
		z = inps[:,2]
		v = inps[:,3]
		a = (np.cos(x*y-(v+2.3*x))**2)[:,np.newaxis]
		return a#np.concatenate((a, 1-a), axis=1)

	def generate_examples(nb_examples):
		inps = np.random.rand(nb_examples,4)
		outs1 = f1(inps)
		outs2 = f2(inps)
		return [[inps]], [[outs1, outs2]]

	examples_train = generate_examples(100000)

	nn.SGD(examples_train, 128, 15, 0.5, 0.5)


	examples_test = generate_examples(10)

	print(nn.get_loss_of_data(examples_train))
	print(nn.get_loss_of_data(examples_test))
	y1, y2 = nn.prop(examples_test[0])[0]

	for i in range(10):
		print(examples_test[0][0][0][i,:],
			"\n\t--> R", examples_test[1][0][0][i,:], "\t    P", y1[i,:],
			'\n\t--> R', examples_test[1][0][1][i,:], "\t    P", y2[i,:])
		print()










