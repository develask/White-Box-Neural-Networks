import numpy as np
import os
import json

class Activation_Function():
	def __init__(self, name):
		self.name = name
		if self.name == "sigmoid":
			def ff_(z):
				return 1 / (1 + np.exp(-z))
			self.ff = ff_
			def derivative_ff_(a):
				return a*(1-a)
			self.derivative_ff = derivative_ff_


		elif self.name == "softmax":
			def ff_ (z):
				exp = np.exp(z)
				return exp/np.sum(exp, axis=1)[:,np.newaxis]
			self.ff = ff_
			def derivative_ff_(a):
				return a*(1-a)
			self.derivative_ff = derivative_ff_

		# elif self.name == "relu":
		# 	self.ff = lambda z: np.maximum(z, 0)
		# 	self.derivative_ff = lambda a: np.sign(a)

		elif self.name == "linear":
			self.ff = lambda z: z
			self.derivative_ff = lambda a: np.full(a.shape, 1)
		else:
			raise ValueError("Not defined activation function")

class Layer:
	def __init__(self, name):
		self.next = []
		self.prev = []
		self.next_rercurrent = []
		self.prev_recurrent = []
		self.initialize_done = False
		self.name = name
		self.a = None
		self.error = None
		self.b_grad = None
		self.W_grad = None
		self.is_in_recurrent_part = False

	def addNext(self, layer):
		self.next.append(layer)
		layer.__addPrev__(self)

	def addNextRecurrent(self, layer):
		self.next_rercurrent.append(layer)
		layer.__addPrevRecurrent__(self)

	def __addPrev__(self, layer):
		self.prev.append(layer)

	def __addPrevRecurrent__(self, layer):
		self.prev_recurrent.append(layer)

	def set_in_recurrent_part(self, par = False):
		self.is_in_recurrent_part = par

	def get_in_recurrent_part(self):
		return self.is_in_recurrent_part

	def get_Output(self, t=0):
		return self.a[t-1]

	def get_error(self, t=0):
		return self.error[t-1]

	def get_Output_dim(self):
		return self.output_dim

	def set_loss_error(self, loss_gradient):
		raise NotImplementedError( "Should have implemented this" )

	def get_error_contribution(self, layer, t=0):
		raise NotImplementedError( "Should have implemented this" )

	def backprop_error(self,t=0):
		raise NotImplementedError( "Should have implemented this" )

	def reset_grads(self):
		raise NotImplementedError( "Should have implemented this" )

	def compute_gradients(self):
		raise NotImplementedError( "Should have implemented this" )

	def apply_to_gradients(self, func):
		raise NotImplementedError( "Should have implemented this" )

	def update(self):
		raise NotImplementedError( "Should have implemented this" )

	def initialize(self):
		raise NotImplementedError( "Should have implemented this" )

	def prop(self, x_labels, proped = []):
		raise NotImplementedError( "Should have implemented this" )

	def reset(self, minibatch_size):
		if len(self.next_rercurrent)>0:
			self.a = [np.zeros((self.output_dim,minibatch_size))]
		else:
			self.a = []
		self.error = []

	def copy(self):
		raise NotImplementedError( "Should have implemented this" )

	def __save__dict__(self):
		raise NotImplementedError( "Should have implemented this" )

	def __load__dict__(self, d):
		raise NotImplementedError( "Should have implemented this" )

	def __dict_maker__(self, d, dir_):
		em = [] if isinstance(d, list) else {}
		if isinstance(d, list):
			for i in range(len(d)):
				if isinstance(d[i], np.ndarray):
					em.append("@"+str(i))
					np.save(dir_+"/@"+str(i), d[i])
				else:
					os.makedirs(dir_+'/'+str(i))
					em.append(self.__dict_maker__(d[i], dir_+'/'+str(i)))
		elif isinstance(d, dict):
			for el in d:
				if isinstance(d[el], np.ndarray):
					em[el] = "@"+str(el)
					np.save(dir_+"/@"+str(el), d[el])
				else:
					os.makedirs(dir_+'/'+str(el))
					em[el] = self.__dict_maker__(d[el], dir_+'/'+str(el))
		else:
			em = d
		return em

	def __dict_load__(d, dir_):
		if isinstance(d, list):
			for i in range(len(d)):
				if isinstance(d[i], str) and d[i][0]=='@':
					d[i] = np.load(dir_+'/'+d[i]+'.npy')
				else:
					d[i] = Layer.__dict_load__(d[i], dir_+'/'+i)
		if isinstance(d, dict):
			for el in d:
				if isinstance(d[el], str) and d[el][0]=='@':
					d[el] = np.load(dir_+'/'+d[el]+'.npy')
				else:
					d[el] = Layer.__dict_load__(d[el], dir_+'/'+el)
		return d

	def save(self, dir_):
		my_id = id(self)
		my_dir = dir_+'/'+str(my_id)
		class_name = self.__class__.__name__
		os.makedirs(my_dir)
		d, ini = self.__save__dict__()
		me = {
			'id': my_id,
			'class': class_name,
			'name': self.name,
			'ini': ini,
			'info': self.__dict_maker__(d, my_dir)
		}
		file_ = json.dumps(me)
		with open(my_dir+"/setup.json", "w") as f:
			f.write(file_)

	def load(dir_):
		with open(dir_+"/setup.json", "r") as f:
			me = json.loads(f.read())
			d = me['info']
			constructor = globals()[me['class']]
			instance = constructor(*me['ini'])
			instance.__load__dict__(Layer.__dict_load__(d, dir_))
			return instance

class Input(Layer):
	def __init__(self, input_dim, name):
		super(Input, self).__init__(name)
		self.input_dim = input_dim

	def get_Output_dim(self):
		return self.input_dim

	def initialize(self):
		self.initialize_done = True

	def compute_gradients(self):
		pass

	def prop(self, x_labels):
		self.a = self.a + [x_labels]
		return self.a

	def copy(self):
		return Input(self.input_dim, self.name)

	def __save__dict__(self):
		return {}, [self.input_dim, self.name]

	def __load__dict__(self, d):
		pass

class Fully_Connected(Layer):
	def __init__(self, output_dim, activation_function, name):
		super(Fully_Connected, self).__init__(name)
		self.output_dim = output_dim
		self.act_f = Activation_Function(activation_function)

	def initialize(self):
		self.W = [np.random.normal(scale=1 ,size=(l.get_Output_dim(), self.get_Output_dim())) for l in self.prev_recurrent + self.prev]
		self.b = np.random.normal(scale=1, size=(1, self.output_dim))
		self.initialize_done = True

	def W_of_layer(self, layer):
		return self.W[(self.prev_recurrent + self.prev).index(layer)]

	def set_loss_error(self, loss_gradient):
		#    This function will be used in the first step of the BP.,
		# when the error is set from the cost function (in supervised learning)
		self.error = [loss_gradient]

	def get_error_contribution(self, layer, t=0):
		return np.dot(
			self.get_error(t=t)*self.act_f.derivative_ff(self.get_Output(t=t)),
			np.transpose(self.W_of_layer(layer))
		)
	def backprop_error(self,t=0):
		#  this function will be used during the BP. to calculate the error
		aux = 0
		if t != 0:
			for layer in self.next_rercurrent:
				aux += layer.get_error_contribution(self, t=t+1)
		for layer in self.next:
			if (t!=0 and layer.get_in_recurrent_part()) or t == 0:
				aux += layer.get_error_contribution(self, t=t)
		self.error = [aux]+self.error

	def reset_grads(self):
		self.b_grad = 0
		self.W_grad = [0]*(len(self.prev)+len(self.prev_recurrent))

	def compute_gradients(self):
		num_t = len(self.error)
		b_grad_aux = 0
		W_grad_aux = [0]*len(self.W_grad)
		for t in range(0,-num_t, -1):
			aux = self.get_error(t=t)*self.act_f.derivative_ff(self.get_Output(t=t))
			b_grad_aux += aux
			W_grad_aux = list(map(lambda x,y: x+y,
				W_grad_aux,
				[np.dot(np.transpose(l.get_Output(t=t-1)), aux) for l in self.prev_recurrent] +
				[np.dot(np.transpose(l.get_Output(t=t)), aux) for l in self.prev]
			))
		self.b_grad += b_grad_aux
		self.W_grad = list(map(lambda x,y: x+y,
				self.W_grad, W_grad_aux
		))


	def apply_to_gradients(self, func):
		self.b_grad = func(self.b_grad)
		self.W_grad = list(map(func, self.W_grad))

	def update(self):
		self.__update_b__()
		self.__update_W__()

	def __update_W__(self):
		for i in range(len(self.W_grad)):
			self.W[i] -= self.W_grad[i]

	def __update_b__(self):
		self.b -= np.sum(self.b_grad, axis=0)[np.newaxis]

	def prop(self):
		inp = 0
		i = 0
		for l in self.prev_recurrent + self.prev:
			inp += np.dot(l.get_Output(), self.W[i])
			i+=1
		self.z = inp + self.b
		out = self.act_f.ff(self.z)
		self.a = self.a + [out]
		return out

	def copy(self):
		return Fully_Connected(self.output_dim, self.act_f.name, self.name)

	def __save__dict__(self):
		return {
			'W': self.W,
			'b': self.b
		}, [self.output_dim, self.act_f.name, self.name]

	def __load__dict__(self, d):
		self.W = d['W']
		self.b = d['b']

class Softmax(Fully_Connected):
	def __init__(self, output_dim, name):
		class Act_func():
			def __init__(self, name):
				self.name = name
			def ff(self, z):
				exp = np.exp(z)
				return exp/np.sum(exp, axis=1)[:,np.newaxis]
			def derivative_ff(self, a):
				def my_func(x):
					x=x[np.newaxis]
					a = -np.dot(x.T,x)
					np.fill_diagonal(a, x*(1-x))
					return a
				b = np.apply_along_axis(my_func, 1, a)
				return b

		activation_function = Act_func("softmax")
		super(Softmax, self).__init__(output_dim, "sigmoid", name)
		self.act_f = activation_function

	def get_error_contribution(self, layer, t=0):
		return np.dot(
			np.einsum('ij,ijk->ik',
				self.get_error(t=t),
				self.act_f.derivative_ff(self.get_Output(t=t))
			),
			np.transpose(self.W_of_layer(layer))
		)

	def compute_gradients(self):
		num_t = len(self.error)
		b_grad_aux = 0
		W_grad_aux = [0]*len(self.W_grad)
		for t in range(0,-num_t, -1):
			b_grad_aux += np.einsum('ij,ijk->ik',
							self.get_error(t=t),
							self.act_f.derivative_ff(self.get_Output(t=t))
							)
			W_grad_aux = list(map(lambda x,y: x+y,
				W_grad_aux,
				[np.dot(np.transpose(l.get_Output(t=t-1)), np.einsum('ij,ijk->ik', self.get_error(t=t),self.act_f.derivative_ff(self.get_Output(t=t)))) for l in self.prev_recurrent] +
				[np.dot(np.transpose(l.get_Output(t=t)), np.einsum('ij,ijk->ik', self.get_error(t=t),self.act_f.derivative_ff(self.get_Output(t=t)))) for l in self.prev]
			))
		self.b_grad += b_grad_aux
		self.W_grad = list(map(lambda x,y: x+y,
				self.W_grad, W_grad_aux
		))

	def copy(self):
		return Softmax(self.output_dim, self.name)

	def __save__dict__(self):
		return {
			'W': self.W,
			'b': self.b
		}, [self.output_dim, self.name]

class LSTM(Layer):
	def __init__(self, num_cel, name):
		super(LSTM, self).__init__(name)
		self.name = name
		self.num_cel = num_cel

	def __sigm__(self, z):
		return 1 / (1 + np.exp(-z))

	def __dev_sigm__(self, a):
		return a*(1-a)

	def __tanh__(self, z):
		return np.tanh(z)

	def __dev_tanh__(self, a):
		return 1 - np.tanh(a)**2

	def get_Output(self, t=0):
		return self.a[t-1]

	def get_i(self, t=0):
		return self.i[t-1]

	def get_f(self, t=0):
		return self.f[t-1]

	def get_c(self, t=0):
		return self.c[t-1]

	def get_o(self, t=0):
		return self.o[t-1]

	def get_a_error(self, t=0):
		return self.error_a[t-1]

	def get_o_error(self, t=0):
		return self.error_o[t-1]

	def get_c_error(self, t=0):
		return self.error_c[t-1]

	def get_f_error(self, t=0):
		return self.error_f[t-1]

	def get_i_error(self, t=0):
		return self.error_i[t-1]
		
	def set_in_recurrent_part(self, par = False):
		pass

	def get_in_recurrent_part(self):
		return True

	def set_loss_error(self, loss_gradient):
		"""this function will be used in the first step of the BP.,
		when the error is set from the cost function (in supervised learning)"""
		self.error_a = [loss_gradient]
		self.backprop_error(t=0)


	def get_error_contribution(self, layer, t=0):
		idx = self.layers.index(layer)
		return np.dot(self.get_o_error(t=t) * self.__dev_sigm__(self.get_o(t=t)), np.transpose(self.W_o_alprevs[idx]))
		+ np.dot(self.get_c_error(t=t) * self.get_i(t=t) * self.__dev_tanh__(self.get_c(t=t)), np.transpose(self.W_c_alprevs[idx]))
		+ np.dot(self.get_i_error(t=t) * self.__dev_sigm__(self.get_i(t=t)), np.transpose(self.W_i_alprevs[idx]))
		+ np.dot(self.get_f_error(t=t) * self.__dev_sigm__(self.get_f(t=t)), np.transpose(self.W_f_alprevs[idx]))

	def backprop_error(self,t=0):
		"""this function will be used during the BP. to calculate the error"""
		aux = 0
		# Other layers' contributions
		if t != 0:
			for layer in self.next_rercurrent:
				aux += layer.get_error_contribution(self, t=t+1)
		for layer in self.next:
			if (t!=0 and layer.get_in_recurrent_part()) or t == 0:
				aux += layer.get_error_contribution(self, t=t)

		if t != 0:
			aux += np.dot(
				self.get_o_error(t=t+1) * self.__dev_sigm__(self.get_o(t=t+1)),
				np.transpose(self.W_o_atprev)
			) + \
			np.dot(
				self.get_c_error(t=t+1) * self.get_i(t=t+1) * self.__dev_tanh__(self.get_c(t=t+1)),
				np.transpose(self.W_c_atprev)
			) + \
			np.dot(
				self.get_i_error(t=t+1) * self.__dev_sigm__(self.get_i(t=t+1)),
				np.transpose(self.W_i_atprev)
			) + \
			np.dot(
				self.get_f_error(t=t+1) * self.__dev_sigm__(self.get_f(t=t+1)),
				np.transpose(self.W_f_atprev)
			)
		
		if isinstance(aux, np.ndarray):
			self.error_a = [aux]+self.error_a

		# error in o
		self.error_o = [self.get_a_error(t=t) * self.__tanh__(self.get_c(t=t))] + self.error_o

		# error in c
		aux = self.w_o_c * self.get_o_error(t=t) * self.__dev_sigm__(self.get_o(t=t)) \
		+ self.get_a_error(t=t) * self.get_o(t=t) * self.__dev_tanh__(self.get_c(t=t))
		if t!=0:
			aux += self.get_c_error(t=t+1) * self.get_f(t=t+1) \
			+ self.w_i_ctprev * self.get_i_error(t=t+1) * self.__dev_sigm__(self.get_i(t=t+1)) \
			+ self.w_f_ctprev * self.get_f_error(t=t+1) * self.__dev_sigm__(self.get_f(t=t+1))
		self.error_c = [aux] + self.error_c

		# error in f
		self.error_f = [self.get_c(t=t-1)*self.get_c_error(t=t)] + self.error_f

		# error in i
		self.error_i = [self.get_c_error(t=t)*self.__tanh__(self.get_c(t=t))] + self.error_i



	def reset_grads(self):
		self.W_i_alprevs_grads = [0]*len(self.layers)
		self.W_f_alprevs_grads = [0]*len(self.layers)
		self.W_c_alprevs_grads = [0]*len(self.layers)
		self.W_o_alprevs_grads = [0]*len(self.layers)

		self.W_i_atprev_grads = 0
		self.W_f_atprev_grads = 0
		self.W_c_atprev_grads = 0
		self.W_o_atprev_grads = 0

		self.w_i_ctprev_grads = 0
		self.w_f_ctprev_grads = 0
		self.w_o_c_grads = 0

		self.b_i_grads = 0
		self.b_f_grads = 0
		self.b_c_grads = 0
		self.b_o_grads = 0

	def compute_gradients(self):
		num_t = len(self.error_a)
		#print(self.error_a)

		W_i_alprevs_grads_aux = [0]*len(self.layers)
		W_f_alprevs_grads_aux = [0]*len(self.layers)
		W_c_alprevs_grads_aux = [0]*len(self.layers)
		W_o_alprevs_grads_aux = [0]*len(self.layers)

		W_i_atprev_grads_aux = 0
		W_f_atprev_grads_aux = 0
		W_c_atprev_grads_aux = 0
		W_o_atprev_grads_aux = 0

		w_i_ctprev_grads_aux = 0
		w_f_ctprev_grads_aux = 0
		w_o_c_grads_aux = 0

		b_i_grads_aux = 0
		b_f_grads_aux = 0
		b_c_grads_aux = 0
		b_o_grads_aux = 0
		for t in range(0,-num_t, -1):
			aux_o = self.get_o_error(t=t) * self.__dev_sigm__(self.get_o(t=t))
			aux_c = self.get_c_error(t=t) * self.get_i(t=t) * self.__dev_tanh__(self.get_c(t=t))
			aux_f = self.get_f_error(t=t) * self.__dev_sigm__(self.get_f(t=t))
			aux_i = self.get_i_error(t=t) * self.__dev_sigm__(self.get_i(t=t))

			for layer in self.layers:
				W_o_alprevs_grads_aux = list(map(lambda x,y: x+y,
					W_o_alprevs_grads_aux,
					[np.dot(np.transpose(l.get_Output(t=t-1)), aux_o) for l in self.prev_recurrent]+\
					[np.dot(np.transpose(l.get_Output(t=t)), aux_o) for l in self.prev]
				))

				W_c_alprevs_grads_aux = list(map(lambda x,y: x+y,
					W_c_alprevs_grads_aux,
					[np.dot(np.transpose(l.get_Output(t=t-1)), aux_c) for l in self.prev_recurrent]+\
					[np.dot(np.transpose(l.get_Output(t=t)), aux_c) for l in self.prev]
				))

				W_f_alprevs_grads_aux = list(map(lambda x,y: x+y,
					W_f_alprevs_grads_aux,
					[np.dot(np.transpose(l.get_Output(t=t-1)), aux_f) for l in self.prev_recurrent]+\
					[np.dot(np.transpose(l.get_Output(t=t)), aux_f) for l in self.prev]
				))

				W_i_alprevs_grads_aux = list(map(lambda x,y: x+y,
					W_i_alprevs_grads_aux,
					[np.dot(np.transpose(l.get_Output(t=t-1)), aux_i) for l in self.prev_recurrent]+\
					[np.dot(np.transpose(l.get_Output(t=t)), aux_i) for l in self.prev]
				))

			W_i_atprev_grads_aux += np.dot(np.transpose(self.get_Output(t=t-1)), aux_i)
			W_f_atprev_grads_aux += np.dot(np.transpose(self.get_Output(t=t-1)), aux_f)
			W_c_atprev_grads_aux += np.dot(np.transpose(self.get_Output(t=t-1)), aux_c)
			#print(self.name, t, W_c_atprev_grads_aux)
			W_o_atprev_grads_aux += np.dot(np.transpose(self.get_Output(t=t-1)), aux_o)

			w_i_ctprev_grads_aux += aux_i * self.get_c(t=t-1)
			w_f_ctprev_grads_aux += aux_f * self.get_c(t=t-1)
			w_o_c_grads_aux += aux_o * self.get_c(t=t)

			b_i_grads_aux += aux_i
			b_f_grads_aux += aux_f
			b_c_grads_aux += aux_c
			b_o_grads_aux += aux_o

		self.W_i_alprevs_grads = [self.W_i_alprevs_grads[i] + W_i_alprevs_grads_aux[i] for i in range(len(W_i_alprevs_grads_aux))]
		self.W_f_alprevs_grads = [self.W_f_alprevs_grads[i] + W_f_alprevs_grads_aux[i] for i in range(len(W_f_alprevs_grads_aux))]
		self.W_c_alprevs_grads = [self.W_c_alprevs_grads[i] + W_c_alprevs_grads_aux[i] for i in range(len(W_c_alprevs_grads_aux))]
		self.W_o_alprevs_grads = [self.W_o_alprevs_grads[i] + W_o_alprevs_grads_aux[i] for i in range(len(W_o_alprevs_grads_aux))]

		self.W_i_atprev_grads += W_i_atprev_grads_aux 
		self.W_f_atprev_grads += W_f_atprev_grads_aux 
		self.W_c_atprev_grads += W_c_atprev_grads_aux 
		self.W_o_atprev_grads += W_o_atprev_grads_aux 

		self.w_i_ctprev_grads += np.sum(w_i_ctprev_grads_aux, axis=0)[np.newaxis] 
		self.w_f_ctprev_grads += np.sum(w_f_ctprev_grads_aux, axis=0)[np.newaxis]
		self.w_o_c_grads += np.sum(w_o_c_grads_aux, axis=0)[np.newaxis]

		self.b_i_grads += np.sum(b_i_grads_aux, axis=0)[np.newaxis]
		self.b_f_grads += np.sum(b_f_grads_aux, axis=0)[np.newaxis]
		self.b_c_grads += np.sum(b_c_grads_aux, axis=0)[np.newaxis]
		self.b_o_grads += np.sum(b_o_grads_aux, axis=0)[np.newaxis]

	def apply_to_gradients(self, func):
		self.W_i_alprevs_grads = list(map(func, self.W_i_alprevs_grads))
		self.W_f_alprevs_grads = list(map(func, self.W_f_alprevs_grads))
		self.W_c_alprevs_grads = list(map(func, self.W_c_alprevs_grads))
		self.W_o_alprevs_grads = list(map(func, self.W_o_alprevs_grads))

		self.W_i_atprev_grads = func(self.W_i_atprev_grads)
		self.W_f_atprev_grads = func(self.W_f_atprev_grads)
		self.W_c_atprev_grads = func(self.W_c_atprev_grads)
		self.W_o_atprev_grads = func(self.W_o_atprev_grads)

		self.w_i_ctprev_grads = func(self.w_i_ctprev_grads)
		self.w_f_ctprev_grads = func(self.w_f_ctprev_grads)
		self.w_o_c_grads = func(self.w_o_c_grads)

		self.b_i_grads = func(self.b_i_grads)
		self.b_f_grads = func(self.b_f_grads)
		self.b_c_grads = func(self.b_c_grads)
		self.b_o_grads = func(self.b_o_grads)

	def update(self):
		for i in range(len(self.W_i_alprevs)):
			self.W_i_alprevs[i] -= self.W_i_alprevs_grads[i]
			self.W_f_alprevs[i] -= self.W_f_alprevs_grads[i]
			self.W_c_alprevs[i] -= self.W_c_alprevs_grads[i]
			self.W_o_alprevs[i] -= self.W_o_alprevs_grads[i]

		self.W_i_atprev -= self.W_i_atprev_grads
		self.W_f_atprev -= self.W_f_atprev_grads
		self.W_c_atprev -= self.W_c_atprev_grads
		self.W_o_atprev -= self.W_o_atprev_grads

		self.w_i_ctprev -= self.w_i_ctprev_grads
		self.w_f_ctprev -= self.w_f_ctprev_grads
		self.w_o_c -= self.w_o_c_grads

		self.b_i -= self.b_i_grads
		self.b_f -= self.b_f_grads
		self.b_c -= self.b_c_grads
		self.b_o -= self.b_o_grads

	def get_Output_dim(self):
		return self.num_cel

	def initialize(self):
		self.layers = self.prev_recurrent + self.prev

		self.W_i_alprevs = [np.random.normal(scale=0.3 ,size=(l.get_Output_dim(), self.get_Output_dim())) for l in self.layers]
		self.W_f_alprevs = [np.random.normal(scale=0.3 ,size=(l.get_Output_dim(), self.get_Output_dim())) for l in self.layers]
		self.W_c_alprevs = [np.random.normal(scale=0.3 ,size=(l.get_Output_dim(), self.get_Output_dim())) for l in self.layers]
		self.W_o_alprevs = [np.random.normal(scale=0.3 ,size=(l.get_Output_dim(), self.get_Output_dim())) for l in self.layers]

		self.W_i_atprev = np.random.normal(scale=0.3 ,size=(self.get_Output_dim(), self.get_Output_dim()))
		self.W_f_atprev = np.random.normal(scale=0.3 ,size=(self.get_Output_dim(), self.get_Output_dim()))
		self.W_c_atprev = np.random.normal(scale=0.3 ,size=(self.get_Output_dim(), self.get_Output_dim()))
		self.W_o_atprev = np.random.normal(scale=0.3 ,size=(self.get_Output_dim(), self.get_Output_dim()))

		self.w_i_ctprev = np.random.normal(scale=0.3 ,size=(1, self.get_Output_dim()))
		self.w_f_ctprev = np.random.normal(scale=0.3 ,size=(1, self.get_Output_dim()))
		self.w_o_c = np.random.normal(scale=0.3 ,size=(1, self.get_Output_dim()))

		self.b_i = np.random.normal(scale=0.3 ,size=(1, self.get_Output_dim()))
		self.b_f = np.random.normal(scale=0.3 ,size=(1, self.get_Output_dim()))
		self.b_c = np.random.normal(scale=0.3 ,size=(1, self.get_Output_dim()))
		self.b_o = np.random.normal(scale=0.3 ,size=(1, self.get_Output_dim()))


	def prop(self):
		len_prev_rec = len(self.prev_recurrent)

		self.i += [None]
		self.f += [None]
		self.c += [None]
		self.o += [None]
		self.a += [None]

		self.i[-1] = self.__sigm__(sum([
				np.dot(self.prev_recurrent[l_i].get_Output(t=-1), self.W_i_alprevs[l_i])
				for l_i in range(len(self.prev_recurrent))
			])\
			+ sum([
				np.dot(self.prev[l_i].get_Output(t=0), self.W_i_alprevs[len_prev_rec+l_i])
					for l_i in range(len(self.prev))
			])\
			+ np.dot(self.get_Output(t=-1), self.W_i_atprev)\
			+ self.w_i_ctprev * self.get_c(t=-1)\
			+ self.b_i
		)

		self.f[-1] = self.__sigm__(sum([
				np.dot(self.prev_recurrent[l_i].get_Output(t=-1), self.W_f_alprevs[l_i])
				for l_i in range(len(self.prev_recurrent))
			])\
			+ sum([
				np.dot(self.prev[l_i].get_Output(t=0), self.W_f_alprevs[len_prev_rec+l_i])
					for l_i in range(len(self.prev))
			])\
			+ np.dot(self.get_Output(t=-1), self.W_f_atprev)\
			+ self.w_f_ctprev * self.get_c(t=-1)\
			+ self.b_f
		)

		self.c[-1] = self.get_f(t=0) * self.get_c(t=-1) \
			+ self.__tanh__(
				sum([
					np.dot(self.prev_recurrent[l_i].get_Output(t=-1), self.W_c_alprevs[l_i])
					for l_i in range(len(self.prev_recurrent))
				])\
				+ sum([
					np.dot(self.prev[l_i].get_Output(t=0), self.W_c_alprevs[len_prev_rec+l_i])
						for l_i in range(len(self.prev))
				])\
				+ np.dot(self.get_Output(t=-1), self.W_c_atprev)\
				+ self.b_c
			)
		self.o[-1] = self.__sigm__(sum([
				np.dot(self.prev_recurrent[l_i].get_Output(t=-1), self.W_o_alprevs[l_i])
				for l_i in range(len(self.prev_recurrent))
			])\
			+ sum([
				np.dot(self.prev[l_i].get_Output(t=0), self.W_o_alprevs[len_prev_rec+l_i])
					for l_i in range(len(self.prev))
			])\
			+ np.dot(self.get_Output(t=-1), self.W_o_atprev)\
			+ self.w_o_c * self.get_c(t=0)\
			+ self.b_o
		)

		self.a[-1] = self.get_o(t=0) * self.__tanh__(self.get_c(t=0))
		return self.get_Output(t=0)

	def reset(self, minibatch_size, init_cond = None):
		if init_cond is None:
			self.a = [np.zeros((minibatch_size, self.num_cel))]
			self.c = [np.zeros((minibatch_size, self.num_cel))]
		else:
			len_ = int(init_cond.shape[0]/2)
			self.a = [init_cond[:len_,:]]
			self.c = [init_cond[len_:,:]]
		self.i = []
		self.f = []
		self.o = []
		self.error_a = []
		self.error_o = []
		self.error_c = []
		self.error_f = []
		self.error_i = []

	def copy(self):
		return LSTM(self.num_cel, self.name)

	def __save__dict__(self):
		return {
			"W_i_alprevs": self.W_i_alprevs,
			"W_f_alprevs": self.W_f_alprevs,
			"W_c_alprevs": self.W_c_alprevs,
			"W_o_alprevs": self.W_o_alprevs,

			"W_i_atprev": self.W_i_atprev,
			"W_f_atprev": self.W_f_atprev,
			"W_c_atprev": self.W_c_atprev,
			"W_o_atprev": self.W_o_atprev,

			"w_i_ctprev": self.w_i_ctprev,
			"w_f_ctprev": self.w_f_ctprev,
			"w_o_c": self.w_o_c,

			"b_i": self.b_i,
			"b_f": self.b_f,
			"b_c": self.b_c,
			"b_o": self.b_o
		}, [self.num_cel, self.name]

	def __load__dict__(self, d):
		self.W_i_alprevs = d["W_i_alprevs"]
		self.W_f_alprevs = d["W_f_alprevs"]
		self.W_c_alprevs = d["W_c_alprevs"]
		self.W_o_alprevs = d["W_o_alprevs"]

		self.W_i_atprev = d["W_i_atprev"]
		self.W_f_atprev = d["W_f_atprev"]
		self.W_c_atprev = d["W_c_atprev"]
		self.W_o_atprev = d["W_o_atprev"]

		self.w_i_ctprev = d["w_i_ctprev"]
		self.w_f_ctprev = d["w_f_ctprev"]
		self.w_o_c = d["w_o_c"]

		self.b_i = d["b_i"]
		self.b_f = d["b_f"]
		self.b_c = d["b_c"]
		self.b_o = d["b_o"]

class Convolution(Layer):
	def __init__(self, shape, kernel, activation_function, name):
		super(Convolution, self).__init__(name)
		assert len(shape) == len(kernel)
		assert sum([shape[s]>=kernel[s] for s in range(len(shape))]) == len(shape)
		assert sum(map(lambda x: x%2!=1, kernel))==0
		self.shape = shape
		self.kernel_shape = kernel
		self.output_dim = np.multiply.reduce(np.subtract(self.shape, self.kernel_shape) +1)
		self.act_f = Activation_Function(activation_function)

	def initialize(self):
		assert sum([l.get_Output_dim() for l in self.prev_recurrent + self.prev]) == np.multiply.reduce(self.shape)
		self.kernel = np.random.normal(scale=1, size=self.kernel_shape)
		# self.W = [np.random.normal(scale=1 ,size=(l.get_Output_dim(), self.get_Output_dim())) for l in self.prev_recurrent + self.prev]
		self.b = np.random.normal(scale=1, size=(1))
		self.initialize_done = True

	def set_loss_error(self, loss_gradient):
		#    This function will be used in the first step of the BP.,
		# when the error is set from the cost function (in supervised learning)
		self.error = [loss_gradient]

	# def get_error_contribution(self, layer, t=0):
	# 	return np.dot(
	# 		self.get_error(t=t)*self.act_f.derivative_ff(self.get_Output(t=t)),
	# 		np.transpose(self.W_of_layer(layer))
	# 	)
	# def backprop_error(self,t=0):
	# 	#  this function will be used during the BP. to calculate the error
	# 	aux = 0
	# 	if t != 0:
	# 		for layer in self.next_rercurrent:
	# 			aux += layer.get_error_contribution(self, t=t+1)
	# 	for layer in self.next:
	# 		if (t!=0 and layer.get_in_recurrent_part()) or t == 0:
	# 			aux += layer.get_error_contribution(self, t=t)
	# 	self.error = [aux]+self.error

	# def reset_grads(self):
	# 	self.b_grad = 0
	# 	self.W_grad = [0]*(len(self.prev)+len(self.prev_recurrent))

	# def compute_gradients(self):
	# 	num_t = len(self.error)
	# 	b_grad_aux = 0
	# 	W_grad_aux = [0]*len(self.W_grad)
	# 	for t in range(0,-num_t, -1):
	# 		aux = self.get_error(t=t)*self.act_f.derivative_ff(self.get_Output(t=t))
	# 		b_grad_aux += aux
	# 		W_grad_aux = list(map(lambda x,y: x+y,
	# 			W_grad_aux,
	# 			[np.dot(np.transpose(l.get_Output(t=t-1)), aux) for l in self.prev_recurrent] +
	# 			[np.dot(np.transpose(l.get_Output(t=t)), aux) for l in self.prev]
	# 		))
	# 	self.b_grad += b_grad_aux
	# 	self.W_grad = list(map(lambda x,y: x+y,
	# 			self.W_grad, W_grad_aux
	# 	))


	# def apply_to_gradients(self, func):
	# 	self.b_grad = func(self.b_grad)
	# 	self.W_grad = list(map(func, self.W_grad))

	def update(self):
		self.__update_b__()
		self.__update_kernel__()

	# def __update_kernel__(self):
	# 	for i in range(len(self.W_grad)):
	# 		self.W[i] -= self.W_grad[i]

	# def __update_b__(self):
	# 	self.b -= np.sum(self.b_grad, axis=0)[np.newaxis]

	def prop(self):
		inp = np.concatenate([l.get_Output() for l in self.prev_recurrent + self.prev], axis=1)
		shape = self.kernel_shape + (inp.shape[0],) + tuple(np.subtract(self.shape, self.kernel_shape) + 1)
		strides = [inp.strides[-1]]
		for sh in reversed(self.shape):
			strides.insert(0, strides[0]*sh)
		strides = (strides * 2)[1:]
		M = np.lib.stride_tricks.as_strided(inp, shape=shape, strides=strides)
		chars = ''.join([chr(97+i) for i in range(len(M.shape))])
		self.z = np.einsum(chars[:len(self.kernel.shape)]+','+chars+'->'+chars[len(self.kernel.shape):], self.kernel, M) + self.b
		out = self.act_f.ff(self.z)
		self.a = self.a + [out]
		return out

	# def copy(self):
	# 	return Fully_Connected(self.output_dim, self.act_f.name, self.name)

	# def __save__dict__(self):
	# 	return {
	# 		'W': self.W,
	# 		'b': self.b
	# 	}, [self.output_dim, self.act_f.name, self.name]

	# def __load__dict__(self, d):
	# 	self.W = d['W']
	# 	self.b = d['b']
		

	

