import dnn
import numpy as np

def extend_object(obj):
	class Recurrent(obj.__class__):
		def get_Output(self, t=0):
			return self.a[t-1]

		def get_error(self, t=0):
			return self.error[t-1]

		def get_b_grad(self, t=0):
			return self.b_grad[t-1]

		def get_W_grad(self, t=0):
			return self.W_grad[t-1]

		def get_mean_b_grad(self, t=0):
			mean_b_grad = None
			for b_grad in self.b_grad:
				if mean_b_grad is None:
					mean_b_grad = b_grad
				else:
					mean_b_grad += b_grad
			return mean_b_grad / len(self.b_grad)

		def get_mean_W_grad(self, t=0):
			mean_W_grads = self.W_grad[0]
			for W_grads in self.W_grad[1:]:
				for i in range(len(W_grads)):
					mean_W_grads[i] += W_grads[i]
			return [mean_W_grad/len(self.W_grad) for mean_W_grad in mean_W_grads]

		def prop(self, *args, **kwargs):
			a_tmp = self.a if self.a else []
			super(obj.__class__, self).prop(*args, **kwargs)
			self.a = a_tmp + [self.a]
			return self.a[-1]

		def set_loss_error(self, loss_gradient):
			tmp = self.error if self.error else []
			self.error = [loss_gradient * self.act_f.derivative_ff(self.get_Output())] + tmp


		def backprop_error(self, t=0):
			tmp = self.error if self.error else []
			wt_d = 0
			for layer in self.next:
				if layer is self:
					if t != 0:
						wt_d += np.dot(np.transpose(layer.W_of_layer(self)), layer.get_error(t=t+1))
				else:
					wt_d += np.dot(np.transpose(layer.W_of_layer(self)), layer.get_error(t=t))
			self.error = [wt_d * self.act_f.derivative_ff(self.get_Output(t=t))] + tmp

		def compute_gradients(self, t=0):
			tmp1 = self.b_grad if self.b_grad else []
			tmp2 = self.W_grad if self.W_grad else []
			self.b_grad = [self.get_error(t=t)] + tmp1
			self.W_grad = [[np.dot(self.b_grad[0], np.transpose(l.get_Output(t= t-1 if l is self else t))) for l in self.prev]] + tmp2

		def reset(self):
			if isinstance(self, Recurrent_Layer):
				self.a = [np.zeros((self.output_dim,1))]
			else:
				self.a = None
			self.error = None
			self.b_grad = None
			self.W_grad = None

	obj.__class__ = Recurrent

class Recurrent_Layer(dnn.Fully_Connected):
	def __init__(self, output_dim, activation_function, name):
		super(Recurrent_Layer, self).__init__(output_dim, activation_function, name)
		self.addNext(self)
		self.a = [np.zeros((output_dim,1))]

class RNN():
	def __init__(self):
		self.dnn = dnn.DNN()

	def initialize(self):
		self.dnn.calculate_layer_order()
		for layer in self.dnn.prop_order:
			if not isinstance(layer, dnn.Input):
				extend_object(layer)
			layer.initialize()
			if isinstance(layer, Recurrent_Layer) and len(layer.next) == 1:
				self.dnn.output_layer = layer
			if len(layer.next) == 0:
				raise Exception('Recurrent Neural Network must end in a recurrent layer.')


	def add_inputs(self, layer):
		self.dnn.add_inputs(layer)

	def get_Output(self):
		return self.dnn.get_Output()

	def get_error(self):
		return self.dnn.get_error()

	def prop(self, inp):
		for layer in self.dnn.prop_order: # falta de añadir
			if not isinstance(layer, dnn.Input):
				layer.reset() # falta de implementar
		out = None
		for inp_t in inp:
			out = self.dnn.prop(inp_t.copy())
		return out

	def backprop(self, inp, desired_output):
		self.prop(inp.copy())

		#backprop para instante t=0
		self.dnn.prop_order[-1].set_loss_error(self.dnn.loss.grad(self.dnn.prop_order[-1].get_Output(t=0), desired_output))
		self.dnn.prop_order[-1].compute_gradients(t=0)

		for layer in self.dnn.prop_order[-2::-1]:
			if not isinstance(layer, dnn.Input):
				layer.backprop_error(t=0)
				layer.compute_gradients(t=0)

		#------------------

		for t in range(-1,-len(inp),-1):
			for layer in self.dnn.prop_order[::-1]:
				if not isinstance(layer, dnn.Input):
					layer.backprop_error(t)
					layer.compute_gradients(t)

	def SGD(self, training_data, batch_size, nb_epochs, lr_start, lr_end):
		lr = lr_start
		dec_rate = (lr_end/lr_start)**(1/(nb_epochs-1))
		W_grads = []
		b_grads = []
		len_tr = len(training_data)

		for i in range(nb_epochs):
			#------------------
			loss = 0
			#out_1 = 0
			for ex in training_data:
				out = self.prop(ex[0].copy())
				loss += self.dnn.loss.ff(out[-1],ex[1])/len_tr

			print("training loss", loss[0])
			#------------------

			random.shuffle(training_data)
			mini_batches = [ training_data[ k:k+batch_size ] for k in range(0, len_tr, batch_size)]
			
			for m_b in mini_batches:
				len_m_b = len(m_b)
				W_grads = [None] * len(self.dnn.prop_order) 
				b_grads = [None] * len(self.dnn.prop_order)
				lr_coef = lr/len_m_b

				# self.backprop(*m_b[0])
				# for k in range(len(self.layers)):
				# 	W_grads.append(self.layers[k].get_W_gradient()*lr/len_m_b)
				# 	b_grads.append(self.layers[k].get_b_gradient()*lr/len_m_b)
				self.backprop(*m_b[0])
				for k in range(len(self.dnn.inputs),len(self.dnn.prop_order)):
					mean_W_grads = self.dnn.prop_order[k].get_mean_W_grad()
					W_grads[k] = [0]*len(mean_W_grads)
					b_grads[k] = 0
					for m in range(len(mean_W_grads)):
						W_grads[k][m] += mean_W_grads[m]*lr_coef
					b_grads[k] += self.dnn.prop_order[k].get_mean_b_grad()*lr_coef

				# for m in range(len(self.prop_order)):
				# 	print("Weight gradient matrix crontibution nb of layer", m, "in example 0 of the minibatch")
				# 	print(-W_grads[m])
				# print("----------------------------------------------")


				for j in range(1,len(m_b)):
					self.backprop(*m_b[j])
					for k in range(len(self.dnn.inputs),len(self.dnn.prop_order)):
						mean_W_grads = self.dnn.prop_order[k].get_mean_W_grad()
						for m in range(len(mean_W_grads)):
							W_grads[k][m] += mean_W_grads[m]*lr_coef
						b_grads[k] += self.dnn.prop_order[k].get_mean_b_grad()*lr_coef

				# ---------- 

				# update weights and biases:
				for k in range(len(self.dnn.inputs),len(self.dnn.prop_order)):
					self.dnn.prop_order[k].update_b(b_grads[k])
					self.dnn.prop_order[k].update_W(W_grads[k])

			lr *= dec_rate



if __name__ == '__main__':
	import random
	import math
	import time
	
	nn = RNN()

	x = dnn.Input(2, "x")
	y = dnn.Input(1, "y")

	def func(x):
		return math.sin(sum([a[1] for a in x]))**2

	R1 = Recurrent_Layer(40, "sigmoid", "r1")
	R2 = Recurrent_Layer(71, "sigmoid", "r2")
	R3 = Recurrent_Layer(1, "sigmoid", "r3")

	ff = dnn.Fully_Connected(83, "sigmoid", "ff")


	x.addNext(R1)
	y.addNext(R2)
	R1.addNext(ff)
	R2.addNext(ff)
	R2.addNext(R3)
	ff.addNext(R3)

	nn.add_inputs(x)
	nn.add_inputs(y)

	nn.initialize()
	training_data = []


	for i in range(20000):
		x = []
		for i2 in range(random.randint(3,5)):
			rr = random.random()
			x.append( [ np.asarray([[1-rr], [rr**2]]), np.asarray([[rr]]) ] )
		training_data.append(
			(
				x,
				np.asarray([[ func(x) ]])
			)
		)
	print("gonna train, holly shit I'm nervous")

	start = time.clock()

	nn.SGD(training_data, 128, 10, 0.5, 0.05)

	end = time.clock()
	interval_c = end-start
	print("Time training: %.2gs" % interval_c)


	for i in range(3):
		x = []
		for i2 in range(random.randint(3,5)):
			rr = random.random()
			x.append( [ np.asarray([[1-rr], [rr**2]]), np.asarray([[rr]]) ] )
		inp = x

		em = func(x)
		print("input:", inp)
		print("expected:", em)
		print("output:", nn.prop(inp))













