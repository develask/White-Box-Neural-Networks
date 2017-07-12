import dnn
import numpy as np

def extend_object(obj):
	class Recurrent_Layer(obj.__class__):
		def get_Output(self, t=0):
			return self.a[t-1]

		def get_error(self, t=0):
			return self.error[t-1]

		def get_b_grad(self, t=0):
			return self.b_grad[t-1]

		def get_W_grad(self, t=0):
			return self.W_grad[t-1]

		def prop(self, *args, **kwargs):
			a_tmp = self.a if self.a else []
			super(obj.__class__, self).prop(*args, **kwargs)
			self.a = a_tmp + [self.a]
			return self.a[-1]

		def set_loss_error(self, loss_gradient, t=0):
			tmp = self.error if self.error else []
			self.error = [loss_gradient * self.act_f.derivative_ff(self.get_Output(t=t))] + tmp

		def backprop_error(self, t=0):
			tmp = self.error if self.error else []
			wt_d = 0
			for layer in self.next:
				wt_d += np.dot(np.transpose(layer.W_of_layer(self)), layer.get_error(t=t))
			self.error = [wt_d * self.act_f.derivative_ff(self.get_Output(t=t))] + tmp

		def compute_gradients(self, t=0):
			tmp1 = self.b_grad if self.b_grad else []
			tmp2 = self.W_grad if self.W_grad else []
			self.b_grad = [self.get_error(t=t)] + tmp1
			self.W_grad = [[np.dot(self.b_grad[0], np.transpose(l.get_Output(t=t))) for l in self.prev]] + tmp2

	obj.__class__ = Recurrent_Layer



class RNN():
	def __init__(self):
		self.dnn = dnn.DNN()
		self.recurrentes = []

	def initialize(self):
		self.dnn.calculate_layer_order()
		for layer in self.dnn.prop_order:
			extend_object(layer)
			layer.initialize()
			if len(layer.next) == 0:
				self.dnn.output_layer = layer

	def add_inputs(self, layer):
		self.dnn.add_inputs(layer)

	def get_Output(self):
		return self.dnn.get_Output()

	def get_error(self):
		return self.dnn.get_error()

	def prop(self, inp):
		for recurrent_layer in self.recurrentes: # falta de añadir
			recurrent_layer.to_initial_conditions() # falta de implementar
		out = None
		for inp_t in inp:
			out = self.dnn.prop(inp_t.copy())
		return out

	def backprop(self, inp, desired_output):
		self.prop(inp.copy())

		self.prop_order[-1].set_loss_error(self.loss.grad(self.prop_order[-1].get_Output(), desired_output))
		self.prop_order[-1].compute_gradients()

		for t in range(0,-len(inp),-1):
			for layer in self.dnn.prop_order[-2 if t==0 else -1:len(self.inputs)-1:-1]:
				if not isinstance(layer, Input):
					layer.backprop_error(t)
					layer.compute_gradients(t)

	def SGD(self, training_data, batch_size, nb_epochs, lr_start, lr_end):
		lr = lr_start
		dec_rate = (lr_end/lr_start)**(1/(nb_epochs-1))
		W_grads = []
		b_grads = []
		len_tr = len(training_data)

		# def porLayer(w_b_layer):
		# 	w = w_b_layer[0] + w_b_layer[2].get_W_gradient()
		# 	b = w_b_layer[1] + w_b_layer[2].get_b_gradient()
		# 	return (w, b)

		# def porEjemplo(W_b, ex):
		# 	self.backprop(*ex)
		# 	return list(zip(*map(porLayer, zip(W_b[0], W_b[1], self.layers))))

		for i in range(nb_epochs):
			#------------------
			loss = 0
			#out_1 = 0
			for ex in training_data:
				out = self.prop(ex[0].copy())
				loss += self.loss.ff(out,ex[1])/len_tr
				#out_1 += out[0][0]/len_tr

			print("training loss", loss[0])
			#------------------
			random.shuffle(training_data)
			mini_batches = [ training_data[ k:k+batch_size ] for k in range(0, len_tr, batch_size)]
			
			for m_b in mini_batches:
				# update the DNN according to this mini batch

				# compute the errors and gradients per example

				# --------

				# W_grads = [0]* len(self.layers)
				# b_grads = [0]* len(self.layers)


				# W_grads, b_grads = reduce(porEjemplo, m_b, (W_grads, b_grads))


				# W_grads = list(map(lambda w: w*lr/len_tr, W_grads))
				# b_grads = list(map(lambda b: b*lr/len_tr, b_grads))


				# -------


				len_m_b = len(m_b)
				W_grads = [None] * len(self.prop_order) 
				b_grads = [None] * len(self.prop_order)
				lr_coef = lr/len_m_b

				# self.backprop(*m_b[0])
				# for k in range(len(self.layers)):
				# 	W_grads.append(self.layers[k].get_W_gradient()*lr/len_m_b)
				# 	b_grads.append(self.layers[k].get_b_gradient()*lr/len_m_b)
				self.backprop(*m_b[0])
				for k in range(len(self.inputs),len(self.prop_order)):
					W_grads[k] = [0]*len(self.prop_order[k].get_W_grad())
					b_grads[k] = 0
					for m in range(len(self.prop_order[k].get_W_grad())):
						W_grads[k][m] += self.prop_order[k].get_W_grad()[m]*lr_coef
					b_grads[k] += self.prop_order[k].get_b_grad()*lr_coef

				# for m in range(len(self.prop_order)):
				# 	print("Weight gradient matrix crontibution nb of layer", m, "in example 0 of the minibatch")
				# 	print(-W_grads[m])
				# print("----------------------------------------------")


				for j in range(1,len(m_b)):
					self.backprop(*m_b[j])
					for k in range(len(self.inputs),len(self.prop_order)):
						for m in range(len(self.prop_order[k].get_W_grad())):
							W_grads[k][m] += self.prop_order[k].get_W_grad()[m]*lr_coef
						b_grads[k] += self.prop_order[k].get_b_grad()*lr_coef

				# ---------- 

				# update weights and biases:
				for k in range(len(self.inputs),len(self.prop_order)):
					self.prop_order[k].update_b(b_grads[k])
					self.prop_order[k].update_W(W_grads[k])

			lr *= dec_rate