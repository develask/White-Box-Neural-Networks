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

	def add_inputs(self, layer):
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

	def get_Output(self):
		return self.output_layer.get_Output()

	def prop(self, inp):
		for layer in self.prop_order:
			layer.reset()
		out = None
		for inp_t in inp:
			inp_t_cop = inp_t.copy()
			out = [layer.prop(inp_t_cop.pop(0)) if isinstance(layer, Input) else layer.prop() for layer in self.prop_order][-1]
		return out

	def backprop(self, inp, desired_output):
		out = self.prop(inp.copy())

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
		len_m_b = len(mini_batch)

		self.reset_gradients()

		for j in range(len(mini_batch)):
			out = self.backprop(*mini_batch[j])
			# print(j,":", out, mini_batch[j][1])
			loss += self.loss.ff(out,mini_batch[j][1])

		# Normalizar los gradientes
		self.apply_to_gradients(lambda grad: grad/len_m_b)

		return loss

	def update_model(self):
		self.apply_to_gradients(lambda x: x*self.lr)
		for k in range(len(self.inputs),len(self.prop_order)):
			self.prop_order[k].update()

	def get_loss_of_data(self, data):
		loss = 0
		len_tr = len(data)
		for ex in data:
			out = self.prop(ex[0].copy())
			loss += self.loss.ff(out,ex[1])
		return loss[0]/len_tr

	def SGD(self, training_data, validation_data, batch_size, nb_epochs, lr_start, lr_end):
		self.lr = lr_start
		dec_rate = (lr_end/lr_start)**(1/(nb_epochs-1))
		len_tr = len(training_data)

		for i in range(nb_epochs):
			loss = 0

			random.shuffle(training_data)
			mini_batches = [ training_data[ k:k+batch_size ] for k in range(0, len_tr, batch_size)]
			
			for m_b in mini_batches:
				loss_m_b = self.get_minibach_grad(m_b)
				loss += loss_m_b
				self.update_model()

			print(i+1,"training loss:", loss[0]/len_tr)
			print("\t\-> validation loss:", self.get_loss_of_data(validation_data))
			self.lr *= dec_rate

		print("final training loss:", self.get_loss_of_data(training_data))
		print("\t\-> validation loss:", self.get_loss_of_data(validation_data))

	def save(self, name = None):
		dir_ = name if name is not None else time.strftime(self.name + "_%Y-%m-%d_%H-%M")
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
	
	nn = DNN("recurrente")

	x = Input(1, "x")

	def func(x):
		return math.sin(sum([a[0] for a in x]))**2

	R1 = Fully_Connected(100, "sigmoid", "r1")
	R2 = Fully_Connected(100, "sigmoid", "r2")
	R3 = Fully_Connected(100, "sigmoid", "r3")

	ff = Fully_Connected(1, "sigmoid", "ff2")

	x.save()
	R1.save()
	R2.save()
	R3.save()
	ff.save()
	quit()


	x.addNext(R1)
	R1.addNext(R2)
	R2.addNext(R3)
	R3.addNext(ff)

	R1.addNextRecurrent(R1)
	R2.addNextRecurrent(R2)
	R3.addNextRecurrent(R3)

	nn.add_inputs(x)

	nn.initialize()
	training_data = []


	for i in range(2000):
		x = []
		for i2 in range(random.randint(3,3)):
			rr = random.random()
			x.append( [ np.asarray([[rr]]) ] )
		training_data.append(
			(
				x,
				np.asarray([[ func(x) ]])
			)
		)
	print("gonna train, holly shit I'm nervous")

	start = time.clock()

	nn.SGD(training_data[500:], training_data[:500], 128, 10, 0.05, 0.005)

	end = time.clock()
	interval_c = end-start
	print("Time training: %.2gs" % interval_c)

	for i in range(10):
		x = []
		for i2 in range(random.randint(3,3)):
			rr = random.random()
			x.append( [ np.asarray([[rr]]) ] )
		inp = x

		em = func(x)

		print("-----------------------")
		#print("input:", inp)
		print("expected:", em)
		print("output:", nn.prop(inp))
