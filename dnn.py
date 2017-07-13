import numpy as np
import random
import math
import time
from functools import reduce


class Activation_Function():
	def __init__(self, name):
		self.name = name
		if self.name == "sigmoid":
			self.ff = lambda z: 1 / (1 + np.exp(-z))
			self.derivate = lambda z: np.exp(-z)/(1+np.exp(-z))**2
			self.derivative_ff = lambda a: a*(1-a)


		elif self.name == "softmax":
			def ff_ (z):
				exp = np.exp(z)
				return exp/np.sum(exp)
			self.ff = ff_
			def derivate_(z):
				exp = np.exp(z)
				suma = np.sum(exp)
				div = exp / suma
				return div - div**2
			self.derivate = derivate_
			self.derivative_ff = lambda a: a - a**2

		elif self.name == "relu":
			self.ff = lambda z: np.maximum(z, 0)
			self.derivate = lambda z: (np.sign(z)+1)/2
			self.derivative_ff = lambda a: np.sign(a)

		elif self.name == "linear":
			self.ff = lambda z: z
			self.derivate = lambda z: np.full(z.shape, 1)
			self.derivative_ff = lambda a: a/a
		else:
			raise ValueError("Not defined activation function")

	# def ff(self, z):
	# 	return self.ff_(z)

	# def derivative(self, z):
	# 	return self.derivate_(z)

	# def derivative_ff(self, a):
	# 	# compute the derivative in function of the ff function, the activation
	# 	return self.derivative_ff_(a)

class Layer:
	def __init__(self, name):
		self.next = []
		self.prev = []
		self.initialize_done = False
		self.name = name
		self.a = None
		self.error = None
		self.b_grad = None
		self.W_grad = None

	def addNext(self, layer):
		self.next.append(layer)
		layer.prev.append(self)

	def get_Input_dim(self):
		return sum(map(lambda l: l.get_Output_dim(), self.prev))

	def get_Output(self, *args, **kargs):
		return self.a

	def get_error(self):
		return self.error

	def get_b_grad(self):
		return self.b_grad

	def get_W_grad(self):
		return self.W_grad

	def set_loss_error(self, loss_gradient):
		# this function will be used in the first step of the BP., when the error is set from 
		# the cost function (in supervised learning)
		self.error = loss_gradient * self.act_f.derivative_ff(self.get_Output())
		#self.error = loss_gradient * self.act_f.derivative(self.z)

	def backprop_error(self):
		wt_d = 0
		for layer in self.next:
			wt_d += np.dot(np.transpose(layer.W_of_layer(self)), layer.get_error())
		self.error = wt_d * self.act_f.derivative_ff(self.get_Output())

	def compute_gradients(self):
		self.b_grad = self.get_error()
		self.W_grad = [np.dot(self.b_grad, np.transpose(l.get_Output())) for l in self.prev]

	def calculate_derivate_ff(self):
		return self.act_f.derivative_ff(self.get_Output())

	def update_W(self, increments):
		for i in range(len(increments)):
			self.W[i] -= increments[i]

	def update_b(self, increment):
		self.b -= increment

	def get_Output_dim(self):
		raise NotImplementedError( "Should have implemented this" )

	def initialize(self):
		raise NotImplementedError( "Should have implemented this" )

	def prop(self, x_labels, proped = []):
		raise NotImplementedError( "Should have implemented this" )

class Input(Layer):
	def __init__(self, input_dim, name):
		super(Input, self).__init__(name)
		self.input_dim = input_dim

	def get_Output_dim(self):
		return self.input_dim

	def initialize(self):
		self.initialize_done = True

	def prop(self, x_labels):
		self.a = x_labels
		return self.a


class Fully_Connected(Layer):
	def __init__(self, output_dim, activation_function, name):
		super(Fully_Connected, self).__init__(name)
		self.output_dim = output_dim
		self.act_f = Activation_Function(activation_function)
		
	def get_Output_dim(self):
		return self.output_dim

	def initialize(self):
		self.W = [np.random.normal(scale=1 ,size=(self.output_dim, l.get_Output_dim())) for l in self.prev]
		self.b = np.zeros((self.output_dim, 1))
		self.initialize_done = True

	def W_of_layer(self, layer):
		for i in range(len(self.prev)):
			if self.prev[i] is layer:
				return self.W[i]

	def prop(self):
		inp = 0
		for i in range(len(self.prev)):
			inp += np.dot(self.W[i], self.prev[i].get_Output())
		self.z = inp + self.b
		self.a = self.act_f.ff(self.z)
		return self.a


# l = Fully_Connected_Layer(3,4, "relu")
# k = l.prop(np.array([[1],[2],[3]]))
# print(k)

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
	def __init__(self):
		self.inputs = []
		self.output_layer = None
		self.loss = Loss("ce1")

	def calculate_layer_order(self):
		self.prop_order = self.inputs.copy()
		i = 0
		while i < len(self.prop_order):
			layer = self.prop_order[i]
			for l in layer.next:
				if sum(map(lambda l_: not l_ in self.prop_order and not l_ is l, l.prev)) == 0 and not l in self.prop_order:
					self.prop_order.append(l)
			i+=1

	# def changeToList(self):
	# 	for layer in self.prop_order:
	# 		prev = []
	# 		for l in layer.prev:
	# 			prev.append(self.prop_order.index(l))
	# 		layer.prev = prev
	# 		next = []
	# 		for l in layer.next:
	# 			next.append(self.prop_order.index(l))
	# 		layer.next = next

	def initialize(self):
		self.calculate_layer_order()
		for layer in self.prop_order:
			layer.initialize()
			if len(layer.next) == 0:
				self.output_layer = layer
		#self.changeToList()

	def add_inputs(self, layer):
		self.inputs.append(layer)

	def get_Output(self):
		return self.output_layer.get_Output()

	def get_error(self):
		return self.output_layer.get_error()

	def prop(self, inp):
		return [layer.prop(inp.pop(0)) if isinstance(layer, Input) else layer.prop() for layer in self.prop_order][-1]

	def backprop(self, inp, desired_output):
		# print(desired_output)
		self.prop(inp.copy())

		#layer_dervs = [l.calculate_derivate_ff() for l in self.prop_order]
		#layer_Ws = [(l.W, l.next_dim, l.next) for l in self.prop_order]

		#errors = [None]*len(layer_dervs)
		# print(self.loss.grad(self.rev_layers[0].get_a_during_prop(), desired_output))
		# print(self.loss.ff(self.rev_layers[0].get_a_during_prop(), desired_output))

		# compute the errors
		self.prop_order[-1].set_loss_error(self.loss.grad(self.prop_order[-1].get_Output(), desired_output))
		self.prop_order[-1].compute_gradients()
		# where rev_layers[0].get_a_during_prop is just the output of the DNN for the given inp.


		#for i in range(len(self.layers)-, -1, -1):
		for layer in self.prop_order[-2:len(self.inputs)-1:-1]:
			if not isinstance(layer, Input):
				layer.backprop_error()
				layer.compute_gradients()
			

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
			# print("Mean output during epoch", i+1,":", out_1)
			# print("------------------------------------------")
			


			# for i in range(len(self.layers)):
			# 	print("Weight matrix of layer nb", i)
			# 	print(self.layers[i].get_W())
			# print("----------------------------------------------")
			# for i in range(len(self.layers)):
			# 	print("Bias matrix of layer nb", i)
			# 	print(self.layers[i].get_b())
			# print("----------------------------------------------")

			# for i in range(len(self.layers)):
			# 	print("Last Weight gradient matrix during training of layer nb", i)
			# 	print(-W_grads[i])
			# print("----------------------------------------------")	
			# for i in range(len(self.layers)):
			# 	print("Last outputs of layer nb", i)
			# 	print(self.layers[i].get_a_during_prop())
			# print("----------------------------------------------")	
			# for i in range(len(self.layers)):
			# 	print("Last z of layer nb", i)
			# 	print(self.layers[i].get_a_during_prop())
			# print("----------------------------------------------")	

		# for i in range(len(self.layers)):
		# 	print("Weight matrix of layer nb", i)
		# 	print(self.layers[i].get_W())
		# print("----------------------------------------------")
		# for i in range(len(self.layers)):
		# 	print("Bias matrix of layer nb", i)
		# 	print(self.layers[i].get_b())
		# print("----------------------------------------------")


					



			#apply_lr_schedule(learning_rate, i)



if __name__ == '__main__':

	nn = DNN()

	x_y = Input(2, "x_y")
	z = Input(1, "z")

	def func(x,y,z):
		return math.sin((z**2)/((x**3)+(y**3))**1/4)**2

	h1 = Fully_Connected(50, "relu", "h1")
	h2 = Fully_Connected(50, "relu", "h2")
	h3 = Fully_Connected(90, "relu", "h3")
	h4 = Fully_Connected(90, "relu", "h4")
	ou = Fully_Connected(2, "softmax", "output")

	x_y.addNext(h1)
	z.addNext(h2)
	h1.addNext(h3)
	h2.addNext(h3)
	h3.addNext(h4)
	h4.addNext(ou)
	

	nn.add_inputs(x_y)
	nn.add_inputs(z)

	nn.initialize()

	# inp = np.asarray([[6],[7],[8],[9],[10],[1],[2],[3],[4],[5]])
	# inp2 = np.asarray([[1],[2],[3],[4],[5]])
	# print("inp:", (inp.shape, inp2.shape))
	# nn.prop([inp, inp2])
	# em = nn.get_Output()
	# print("out:", em.shape)
	# print(em)

	# import time

	# dnn = DNN([Fully_Connected_Layer(1,50, "sigmoid"),
	# 		   Fully_Connected_Layer(50,50, "sigmoid"),
	# 		   Fully_Connected_Layer(50,50, "sigmoid"),
	# 		   Fully_Connected_Layer(50,50, "sigmoid"),
	# 		   Fully_Connected_Layer(50,2, "softmax")])


	### generate some data...

	training_data = []


	for i in range(20000):
		x = random.random()
		y = random.random()
		z = random.random()
		training_data.append(
			(
				[ np.asarray([[x], [y]]) , np.asarray([[z]]) ],
				np.asarray([[ func(x,y,z) ], [ 1 - func(x,y,z) ]])
			)
		)
	print("gonna train, holly shit I'm nervous")

	start = time.clock()

	nn.SGD(training_data, 128, 10, 0.005, 0.005)

	end = time.clock()
	interval_c = end-start
	print("Time training: %.2gs" % interval_c)


	for i in range(3):
		x = random.random()
		y = random.random()
		z = random.random()
		inp = [ np.asarray([[x], [y]]) , np.asarray([[z]])]
		em = np.asarray([[ func(x,y,z) ], [ 1 - func(x,y,z) ]])
		print("input:", inp)
		print("expected:", em)
		print("output:", nn.prop(inp))


	# print("let's compute  sin(x = 0.3)**2 = 0.087")
	# print(nn.prop([np.asarray([[0.3]])]))

	# print("let's compute sin(x = 0.51)**2 = 0.238")
	# print(nn.prop([np.asarray([[0.51]])]))

	# print("let's compute sin(x = 0.9)**2 = 0.614")
	# print(nn.prop([np.asarray([[0.9]])]))






