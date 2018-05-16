import numpy as np
from . import initializers
import os
import json

class Layer:
	def __init__(self, name):
		self.next = []
		self.prev = []
		self.next_recurrent = []
		self.prev_recurrent = []
		self.initialize_done = False
		self.name = name
		self.a = None
		self.error = None
		self.b_grad = None
		self.W_grad = None
		self.prop_idx = 0
		self.initializer = initializers.RandomNormal()

	def setInitializer(self, initializer):
		assert isinstance(initializer, initializers.Initializer)
		self.initializer = initializer

	def addNext(self, layer):
		self.next.append(layer)
		layer.__addPrev__(self)

	def addNextRecurrent(self, layer):
		self.next_recurrent.append(layer)
		layer.__addPrevRecurrent__(self)

	def __addPrev__(self, layer):
		self.prev.append(layer)

	def __addPrevRecurrent__(self, layer):
		self.prev_recurrent.append(layer)

	def get_Output(self, t=0):
		return self.a[t-1]

	def get_error(self, t=0):
		return self.error[t-1]

	def get_Output_dim(self):
		return self.output_dim

	def get_error_contribution(self, layer, t=0):
		raise NotImplementedError( "Should have implemented this" )

	def backprop_error(self,t=0, external_error=0):
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
		if len(self.next_recurrent)>0:
			self.a = [np.zeros((minibatch_size, self.get_Output_dim()))]
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
		return x_labels

	def copy(self):
		return Input(self.input_dim, self.name)

	def __save__dict__(self):
		return {}, [self.input_dim, self.name]

	def __load__dict__(self, d):
		pass

class Fully_Connected(Layer):
	def __init__(self, output_dim, name):
		super(Fully_Connected, self).__init__(name)
		self.output_dim = output_dim

	def initialize(self):
		self.W = [self.initializer.get((l.get_Output_dim(), self.get_Output_dim())) for l in self.prev_recurrent + self.prev]
		self.b = self.initializer.get((1, self.output_dim))
		self.initialize_done = True

	def __W_of_layer__(self, layer):
		return self.W[(self.prev_recurrent + self.prev).index(layer)]

	def get_error_contribution(self, layer, t=0):
		return np.dot(
			self.get_error(t=t),
			np.transpose(self.__W_of_layer__(layer))
		)

	def backprop_error(self, t=0):
		#  this function will be used during the BP. to calculate the error
		aux = 0
		if t != 0:
			for layer in self.next_recurrent:
				aux += layer.get_error_contribution(self, t=t+1)
		for layer in self.next:
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
			aux = self.get_error(t=t)
			if aux is not 0:
				b_grad_aux += np.sum(aux, axis=0)
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
		self.b -= self.b_grad

	def prop(self):
		inp = 0
		i = 0
		for l in self.prev_recurrent:
			inp += np.dot(l.get_Output(t= 0 if l.prop_idx >= self.prop_idx else -1), self.W[i])
			i+=1
		for l in self.prev:
			inp += np.dot(l.get_Output(t=0), self.W[i])
			i+=1
		out = inp + self.b
		self.a = self.a + [out]
		return out

	def copy(self):
		return Fully_Connected(self.output_dim, self.name)

	def __save__dict__(self):
		return {
			'W': self.W,
			'b': self.b
		}, [self.output_dim, self.name]

	def __load__dict__(self, d):
		self.W = d['W']
		self.b = d['b']

class LSTM(Layer):
	def __init__(self, num_cel, name):
		super(LSTM, self).__init__(name)
		self.name = name
		self.num_cel = num_cel

	def __sigm__(self, z):
		return (np.tanh(z/2)+1)/2

	def __dev_sigm__(self, a):
		return a*(1-a)

	def __tanh__(self, z):
		return np.tanh(z)

	def __dev_tanh__(self, a):
		return 1 - a**2

	def __dev_tanh_z__(self, z):
		return 1 - np.tanh(z)**2

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

	def get_tanh_zc(self, t=0):
		return self.tanh_zc[t-1]

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

	def get_error_contribution(self, layer, t=0):
		idx = self.layers.index(layer)
		return np.dot(self.get_o_error(t=t) * self.__dev_sigm__(self.get_o(t=t)), np.transpose(self.W_o_alprevs[idx]))
		+ np.dot(self.get_c_error(t=t) * self.get_i(t=t) * self.__dev_tanh__(self.get_c(t=t)), np.transpose(self.W_c_alprevs[idx]))
		+ np.dot(self.get_i_error(t=t) * self.__dev_sigm__(self.get_i(t=t)), np.transpose(self.W_i_alprevs[idx]))
		+ np.dot(self.get_f_error(t=t) * self.__dev_sigm__(self.get_f(t=t)), np.transpose(self.W_f_alprevs[idx]))

	def backprop_error(self,t=0, external_error=0):
		"""this function will be used during the BP. to calculate the error"""
		aux = external_error
		# Other layers' contributions
		if t != 0:
			for layer in self.next_recurrent:
				aux += layer.get_error_contribution(self, t=t+1)
		for layer in self.next:
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
		+ self.get_a_error(t=t) * self.get_o(t=t) * self.__dev_tanh_z__(self.get_c(t=t))
		if t!=0:
			aux += self.get_c_error(t=t+1) * self.get_f(t=t+1) \
			+ self.w_i_ctprev * self.get_i_error(t=t+1) * self.__dev_sigm__(self.get_i(t=t+1)) \
			+ self.w_f_ctprev * self.get_f_error(t=t+1) * self.__dev_sigm__(self.get_f(t=t+1))
		self.error_c = [aux] + self.error_c

		# error in f
		self.error_f = [self.get_c(t=t-1)*self.get_c_error(t=t)] + self.error_f

		# error in i
		self.error_i = [self.get_c_error(t=t)*self.get_tanh_zc(t=t)] + self.error_i

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

		self.W_i_alprevs = [self.initializer.get((l.get_Output_dim(), self.get_Output_dim())) for l in self.layers]
		self.W_f_alprevs = [self.initializer.get((l.get_Output_dim(), self.get_Output_dim())) for l in self.layers]
		self.W_c_alprevs = [self.initializer.get((l.get_Output_dim(), self.get_Output_dim())) for l in self.layers]
		self.W_o_alprevs = [self.initializer.get((l.get_Output_dim(), self.get_Output_dim())) for l in self.layers]

		self.W_i_atprev = self.initializer.get((self.get_Output_dim(), self.get_Output_dim()))
		self.W_f_atprev = self.initializer.get((self.get_Output_dim(), self.get_Output_dim()))
		self.W_c_atprev = self.initializer.get((self.get_Output_dim(), self.get_Output_dim()))
		self.W_o_atprev = self.initializer.get((self.get_Output_dim(), self.get_Output_dim()))

		self.w_i_ctprev = self.initializer.get((1, self.get_Output_dim()))
		self.w_f_ctprev = self.initializer.get((1, self.get_Output_dim()))
		self.w_o_c = self.initializer.get((1, self.get_Output_dim()))

		self.b_i = self.initializer.get((1, self.get_Output_dim()))
		self.b_f = self.initializer.get((1, self.get_Output_dim()))
		self.b_c = self.initializer.get((1, self.get_Output_dim()))
		self.b_o = self.initializer.get((1, self.get_Output_dim()))

	def prop(self):
		len_prev_rec = len(self.prev_recurrent)

		self.i += [None]
		self.f += [None]
		self.tanh_zc += [None]
		self.c += [None]
		self.o += [None]
		self.a += [None]

		self.i[-1] = self.__sigm__(sum([
				np.dot(self.prev_recurrent[l_i].get_Output(t=0 if self.prev_recurrent[l_i].prop_idx >= self.prop_idx else -1), self.W_i_alprevs[l_i])
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
				np.dot(self.prev_recurrent[l_i].get_Output(t=0 if self.prev_recurrent[l_i].prop_idx >= self.prop_idx else -1), self.W_f_alprevs[l_i])
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

		self.tanh_zc[-1] = self.__tanh__(
			sum([
				np.dot(self.prev_recurrent[l_i].get_Output(t=0 if self.prev_recurrent[l_i].prop_idx >= self.prop_idx else -1), self.W_c_alprevs[l_i])
				for l_i in range(len(self.prev_recurrent))
			])\
			+ sum([
				np.dot(self.prev[l_i].get_Output(t=0), self.W_c_alprevs[len_prev_rec+l_i])
					for l_i in range(len(self.prev))
			])\
			+ np.dot(self.get_Output(t=-1), self.W_c_atprev)\
			+ self.b_c
		)

		self.c[-1] = self.get_f(t=0) * self.get_c(t=-1) + self.get_i(t=0) * self.get_tanh_zc(t=0)

		self.o[-1] = self.__sigm__(sum([
				np.dot(self.prev_recurrent[l_i].get_Output(t=0 if self.prev_recurrent[l_i].prop_idx >= self.prop_idx else -1), self.W_o_alprevs[l_i])
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
		self.tanh_zc = []
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
	def __init__(self, shape, kernel, step, nb_filters, name):
		super(Convolution, self).__init__(name)
		assert len(shape) == len(kernel)
		assert len(shape) == len(step)
		assert sum([shape[s]>=kernel[s] for s in range(len(shape))]) == len(shape)
		# assert sum(map(lambda x: x%2!=1, kernel))==0
		self.shape = tuple(shape)
		self.kernel_shape = (nb_filters,)+tuple(kernel)
		self.step = tuple(step)
		self.nb_filters = nb_filters
		self.output_dim = np.multiply.reduce((np.subtract(self.shape, self.kernel_shape[1:])//self.step)+1) * self.nb_filters

	def initialize(self):
		assert sum([l.get_Output_dim() for l in self.prev_recurrent + self.prev]) == np.multiply.reduce(self.shape)
		assert len(self.prev_recurrent + self.prev)==1 or len(self.prev_recurrent + self.prev) == self.shape[0]
		assert sum([(self.prev_recurrent + self.prev)[i].get_Output_dim() != (self.prev_recurrent + self.prev)[i+1].get_Output_dim() for i in range(len(self.prev_recurrent + self.prev)-1)]) == 0
		self.W = self.initializer.get(self.kernel_shape)
		self.b = self.initializer.get((self.nb_filters,))
		self.initialize_done = True

	def get_error(self, t=0):
		return self.error[t-1].reshape((self.error[t-1].shape[0], self.nb_filters)+tuple((np.subtract(self.shape, self.kernel_shape[1:])//self.step)+1))

	def get_error_contribution(self, layer, t=0):
		# dimension = loss_gradient.shape[1]//self.nb_filters
		# self.error.append([loss_gradient[:,dimension*i:dimension*(i+1)] for i in range(self.nb_filters)])
		if len(self.prev+self.prev_recurrent) != 1:
			inp_shape = tuple(self.shape[1:])
			kernel_shape = tuple(self.kernel_shape[2:])
			index_input = (self.prev_recurrent+self.prev).index(layer)
			pad = ((0,0), (0,0), (0,0))+tuple([(x,x)for x in np.subtract(kernel_shape, 1)])
		else:
			inp_shape = self.shape
			kernel_shape = self.kernel_shape[1:]
			pad = ((0,0), (0,0))+tuple([(x,x)for x in np.subtract(kernel_shape, 1)])

		kk = np.zeros((self.x_M.shape[0],self.nb_filters)+tuple(np.subtract(self.shape, self.kernel_shape[1:])+1))
		kk[[slice(None), slice(None)]+[ slice(None,None,x) for x in self.step]] = self.get_error(t=t)
		mat = np.lib.pad(kk, pad, 'constant', constant_values=0)
		if len(self.prev+self.prev_recurrent) != 1:
			mat = mat[:,:,0,...]
		for i in range(len(kernel_shape)):
			mat = np.flip(mat,axis=i+2)
		shape = (mat.shape[0], mat.shape[1]) + kernel_shape + inp_shape
		strides = tuple([mat.strides[0], mat.strides[1]]) +  mat.strides[2:] + mat.strides[2:]
		mat = np.lib.stride_tricks.as_strided(mat, shape=shape, strides=strides)
		for i in range(len(kernel_shape)):
			mat = np.flip(mat,axis=i+2+len(kernel_shape))
		chars = ''.join([chr(97+i) for i in range(len(mat.shape))])
		if len(self.prev+self.prev_recurrent) != 1:
			result = np.einsum(chars[1:len(self.W.shape[1:])+1]+','+chars+'->a'+chars[2+len(self.W.shape[2:]):], self.W[:,index_input,...], mat)
		else:
			result = np.einsum(chars[1:len(self.W.shape)+1]+','+chars+'->a'+chars[1+len(self.W.shape):], self.W, mat)
		return result.reshape(result.shape[0], layer.get_Output_dim())

	def backprop_error(self,t=0):
		aux = 0
		if t != 0:
			for layer in self.next_recurrent:
				aux += layer.get_error_contribution(self, t=t+1)
		for layer in self.next:
			aux += layer.get_error_contribution(self, t=t)
		self.error = [aux]+self.error

	def reset_grads(self):
		self.b_grad = 0
		self.W_grad = 0

	def compute_gradients(self):
		num_t = len(self.error)
		self.b_grad = np.asarray([sum([np.sum(self.get_error(t=t)[:,f,...]) for t in range(0,-num_t, -1)]) for f in range(self.nb_filters)])
		chars = ''.join([chr(97+i) for i in range(len(self.x_M.shape))])
		for t in range(0,-num_t, -1):
			self.W_grad += np.einsum('aF'+chars[1+len(self.kernel_shape[1:]):]+','+chars+'->F'+chars[1:len(self.kernel_shape[1:])+1], self.get_error(t=t), self.x_M)

	def apply_to_gradients(self, func):
		self.b_grad = func(self.b_grad)
		self.W_grad = func(self.W_grad)

	def update(self):
		self.__update_b__()
		self.__update_W__()

	def __update_W__(self):
		self.W -= self.W_grad

	def __update_b__(self):
		self.b -= self.b_grad

	def prop(self):
		inp = np.concatenate(
			[l.get_Output(t=0 if l.prop_idx >= self.prop_idx else -1) for l in self.prev_recurrent]+
			[l.get_Output(t=0) for l in self.prev], axis=1)
		shape = (inp.shape[0],) + self.kernel_shape[1:] + tuple((np.subtract(self.shape, self.kernel_shape[1:])//self.step)+1)
		strides = [inp.strides[-1]]
		for sh in reversed(self.shape):
			strides.insert(0, strides[0]*sh)
		strides = (strides[0],) + tuple(strides[1:]) + tuple(np.multiply(strides[1:], self.step))
		self.x_M = np.lib.stride_tricks.as_strided(inp, shape=shape, strides=strides)
		chars = ''.join([chr(97+i) for i in range(len(self.x_M.shape))])
		out = np.einsum('L'+chars[1:len(self.kernel_shape[1:])+1]+','+chars+'->aL'+chars[1+len(self.kernel_shape[1:]):], self.W, self.x_M)
		for i in range(self.nb_filters):
			out[:,i,...] += self.b[i]
		out = out.reshape(out.shape[0], self.output_dim)
		self.a = self.a + [out]
		return out

	def copy(self):
		return Convolution(self.shape, self.kernel_shape[1:], self.step, self.nb_filters, self.name)

	def __save__dict__(self):
		return {
			'W': self.W,
			'b': self.b
		}, [self.shape, self.kernel_shape[1:], self.step, self.nb_filters, self.name]

	def __load__dict__(self, d):
		self.W = d['W']
		self.b = d['b']

class Activation(Layer):
	def __init__(self, activation_function, name):
		super(Activation, self).__init__(name)
		self.set_activation_function(activation_function)

	def set_activation_function(self, activation_function):
		self.act_fun = activation_function

		if activation_function == "sigmoid":
			self.ff = lambda z: (np.tanh(z/2)+1)/2
			self.derivative_ff = lambda a: a*(1-a)
			self.multiplication = lambda a,b: a*b

		elif activation_function == "tanh":
			self.ff = lambda z: np.tanh(z)
			self.derivative_ff = lambda a: 1 - a**2
			self.multiplication = lambda a,b: a*b

		elif activation_function == "dummy_softmax":
			def ff_ (z):
				exp = np.exp(z)
				return exp/np.sum(exp, axis=1)[:,np.newaxis]
			self.ff = ff_
			self.derivative_ff = lambda a: a*(1-a)
			self.multiplication = lambda a,b: a*b

		elif activation_function == "softmax":
			def ff(z):
				exp = np.exp(z)
				return exp/np.sum(exp, axis=1)[:,np.newaxis]
			self.ff = ff
			def softmax_calc(x):
				x=x[np.newaxis]
				a = -np.dot(x.T,x)
				np.fill_diagonal(a, x*(1-x))
				return a
			self.softmax_calc = softmax_calc
			def derivative_ff(a):
				b = np.apply_along_axis(self.softmax_calc, 1, a)
				return b
			self.derivative_ff = derivative_ff
			self.multiplication = lambda a,b: np.einsum('ij,ijk->ik', a, b)

		elif activation_function == "relu":
			self.ff = lambda z: np.maximum(z, 0)
			self.derivative_ff = lambda a: np.sign(a)
			self.multiplication = lambda a,b: a*b

		elif activation_function == "leaky_relu":
			self.ff = lambda z: np.maximum(z, 0)
			self.derivative_ff = lambda a: np.maximum(np.sign(a), 0.01)
			self.multiplication = lambda a,b: a*b

		elif activation_function == "linear":
			self.ff = lambda z: z
			self.derivative_ff = lambda a: np.full(a.shape, 1)
			self.multiplication = lambda a,b: a*b
		else:
			raise ValueError("Activation function not defined")

	def initialize(self):
		self.initialize_done = True

	def get_Output_dim(self):
		return sum([l.get_Output_dim() for l in (self.prev_recurrent+self.prev)])

	def get_data_of_layer(self, x, layer):
		i = (self.prev_recurrent+self.prev).index(layer)
		init = sum([l.get_Output_dim() for l in (self.prev_recurrent+self.prev)[:i]])
		end = init+layer.get_Output_dim()
		return x[:,init:end]

	def get_error_contribution(self, layer, t=0):
		return self.get_data_of_layer(self.multiplication(
					self.get_error(t=t),
					self.derivative_ff(self.get_Output(t=t))
				), layer)

	def backprop_error(self,t=0):
		#  this function will be used during the BP. to calculate the error
		aux = 0
		if t != 0:
			for layer in self.next_recurrent:
				aux += layer.get_error_contribution(self, t=t+1)
		for layer in self.next:
			aux += layer.get_error_contribution(self, t=t)
		self.error = [aux]+self.error

	def reset_grads(self):
		pass

	def compute_gradients(self):
		pass

	def apply_to_gradients(self, func):
		pass

	def update(self):
		pass

	def prop(self):
		out = self.ff(np.concatenate([l.get_Output() for l in self.prev_recurrent + self.prev], axis=0))
		self.a = self.a + [out]
		return out

	def copy(self):
		return Activation_Layer(self.act_fun, self.name)

	def __save__dict__(self):
		return {}, [self.act_fun, self.name]

	def __load__dict__(self, d):
		pass

class Loss(Layer):
	def __init__(self, loss, name):
		super(Loss, self).__init__(name)
		self.loss = loss
		self.output_dim = sum([l.get_Output_dim() for l in self.prev_recurrent + self.prev])

	def ff(self, a, y):
		if self.loss == "mse":
			return sum((y-a)**2)/2

		elif self.loss == "ce1":
			return -(y*np.log(a+0.000001))-((1-y)*np.log(1.000001-a))

		elif self.loss == "ce2":
			return -(y*np.log(a+0.000001))

		elif self.loss == "external":
			return a

	def grad(self, a, y):
		# returns the gradient with respect to all the components of a.
		# Thus the dimension of the output np.array is the same that a, i.e. [output_dim, 1]
		if self.loss == "mse":
			return a - y

		elif self.loss == "ce1":
			if np.sum(a == 0) == 0 and np.sum(a==1) == 0:
				# if np.isnan(-y/(a) + (1-y)/(1-a)).any():
				# 	pass
				return -(y/(a)) + (1-y)/(1-a)
			else:
				return -(y/(a+0.000001)) + (1-y)/(1.000001-a)

		elif self.loss == "ce2":
			if np.sum(a == 0) == 0:
				# if np.isnan(-y/(a) + (1-y)/(1-a)).any():
				# 	pass
				return -(y/(a))
			else:
				return -(y/(a+0.000001))

		elif self.loss == "external":
			return y

	def addNext(self, layer):
		assert False, "Loss layer do not have next layers"

	def addNextRecurrent(self, layer):
		assert False, "Loss layer do not have next layers"

	def __addPrev__(self, layer):
		self.prev.append(layer)
		assert len(self.prev) == 1

	def __addPrevRecurrent__(self, layer):
		assert False, "Loss layer do not have recurrent layer prev"

	def get_Output(self, t=0):
		return self.a[t-1]

	def get_error(self, t=0):
		assert False, "no error in loss layers"

	def get_error_contribution(self, layer, t=0):
		if self.desired_output is not None:
			return self.grad(self.prev[0].get_Output(t=t), self.desired_output)
		else:
			return np.zeros(self.prev[0].get_Output(t=t).shape)
	def backprop_error(self,t=0, desired_output=None):
		self.desired_output = desired_output

	def reset_grads(self):
		pass

	def compute_gradients(self):
		pass

	def apply_to_gradients(self, func):
		pass

	def update(self):
		pass

	def initialize(self):
		pass

	def prop(self, desired_output=None):
		out = self.prev[0].get_Output()
		if desired_output is not None:
			out = self.ff(out, desired_output)
		self.a = self.a + [out]
		return out

	def copy(self):
		return Loss(self.loss, self.name)

	def __save__dict__(self):
		return {}, [self.loss, self.name]

	def __load__dict__(self, d):
		pass

class Dropout(Layer):
	def __init__(self, probability, name):
		super(Dropout, self).__init__(name)
		self.probability = probability
		self.output_dim = None

	def get_Output_dim(self):
		if self.output_dim == None:
			self.output_dim = sum([l.get_Output_dim() for l in (self.prev_recurrent+self.prev)])
		return self.output_dim

	def get_data_of_layer(self, x, layer):
		i = (self.prev_recurrent+self.prev).index(layer)
		init = sum([l.get_Output_dim() for l in (self.prev_recurrent+self.prev)[:i]])
		end = init+layer.get_Output_dim()
		return x[:,init:end]

	def get_error_contribution(self, layer, t=0):
		return self.get_data_of_layer(self.get_error(t=t),layer)*self.get_data_of_layer(self.get_Output(t=t),layer)

	def backprop_error(self,t=0):
		#  this function will be used during the BP. to calculate the error
		aux = 0
		if t != 0:
			for layer in self.next_recurrent:
				aux += layer.get_error_contribution(self, t=t+1)
		for layer in self.next:
			aux += layer.get_error_contribution(self, t=t)
		aux = aux * self.mask
		self.error = [aux]+self.error

	def reset(self, minibatch_size):
		if len(self.next_recurrent)>0:
			self.a = [np.zeros((minibatch_size, self.get_Output_dim()))]
		else:
			self.a = []
		self.mask = np.random.rand(minibatch_size, self.get_Output_dim()) > self.probability
		self.error = []

	def reset_grads(self):
		pass

	def compute_gradients(self):
		pass

	def apply_to_gradients(self, func):
		pass

	def update(self):
		pass

	def initialize(self):
		pass

	def prop(self, desired_output=None):
		out = np.concatenate([l.get_Output(t=0 if l.prop_idx >= self.prop_idx else -1) for l in  self.prev_recurrent]+[l.get_Output(t=0) for l in self.prev], axis = 1) * self.mask
		self.a = self.a + [out]
		return out

	def copy(self):
		return Dropout(self.probability, self.name)

	def __save__dict__(self):
		return {}, [self.probability, self.name]

	def __load__dict__(self, d):
		pass

class MaxPooling(Layer):
	def __init__(self, shape, kernel, name):
		super(MaxPooling, self).__init__(name)
		assert len(shape) == len(kernel)
		assert sum([shape[s]>=kernel[s] for s in range(len(shape))]) == len(shape)
		self.shape = tuple(shape)
		self.kernel_shape = tuple(kernel)
		self.output_dim = np.multiply.reduce((np.subtract(self.shape, self.kernel_shape)//self.kernel_shape)+1)

	def get_Output_dim(self):
		return self.output_dim

	def get_data_of_layer(self, x, layer):
		i = (self.prev_recurrent+self.prev).index(layer)
		init = sum([l.get_Output_dim() for l in (self.prev_recurrent+self.prev)[:i]])
		end = init+layer.get_Output_dim()
		return x[:,init:end]

	def get_error_contribution(self, layer, t=0):
		return self.get_data_of_layer(self.get_error(t=t),layer)#*self.get_data_of_layer(self.get_Output(t=t),layer)

	def __reshape_idex__(self, k):
		i = np.arange(0,self.shape[k]-self.kernel_shape[k]+1,self.kernel_shape[k])
		i = np.expand_dims(i, axis=0)
		for d in range(len(self.kernel_shape)):
			if d < k:
				i = np.expand_dims(i, axis=0)
			elif d > k:
				i = np.expand_dims(i, axis=-1)
		return i



	def backprop_error(self,t=0):
		#  this function will be used during the BP. to calculate the error
		aux = 0
		if t != 0:
			for layer in self.next_recurrent:
				aux += layer.get_error_contribution(self, t=t+1)
		for layer in self.next:
			aux += layer.get_error_contribution(self, t=t)
		# aux = aux * self.mask

		d = np.zeros((self.get_Output(t=t).shape[0],)+self.shape)

		i = []
		for k in range(len(self.kernel_shape)):
			i.append((self.masks[t-1][k]+self.__reshape_idex__(k)).flatten())

		if aux is not 0:
			i.insert(0, np.repeat(np.arange(aux.shape[0]), i[0].shape[0]//aux.shape[0]))
			d[i] = aux.flatten()

		aux = d.reshape(d.shape[0], np.multiply.reduce(d.shape[1:]))

		self.error = [aux]+self.error

	def reset(self, minibatch_size):
		if len(self.next_recurrent)>0:
			self.a = [np.zeros((minibatch_size, self.get_Output_dim()))]
		else:
			self.a = []
		self.masks = []
		self.error = []

	def reset_grads(self):
		pass

	def compute_gradients(self):
		pass

	def apply_to_gradients(self, func):
		pass

	def update(self):
		pass

	def initialize(self):
		assert sum([l.get_Output_dim() for l in self.prev_recurrent + self.prev]) == np.multiply.reduce(self.shape), "%s: %s --> %d" % (self.name, self.shape, sum([l.get_Output_dim() for l in self.prev_recurrent + self.prev]))
		assert sum([(self.prev_recurrent + self.prev)[i].get_Output_dim() != (self.prev_recurrent + self.prev)[i+1].get_Output_dim() for i in range(len(self.prev_recurrent + self.prev)-1)]) == 0

	def prop(self, desired_output=None):
		out = np.concatenate([l.get_Output(t=0 if l.prop_idx >= self.prop_idx else -1) for l in self.prev_recurrent]+[l.get_Output(t=0) for l in self.prev], axis = 1)

		shape_ = (out.shape[0],) + tuple((np.subtract(self.shape, self.kernel_shape)//self.kernel_shape)+1) + self.kernel_shape
		strides = [out.strides[-1]]
		for sh in reversed(self.shape):
			strides.insert(0, strides[0]*sh)
		strides = (strides[0],) + tuple(np.multiply(strides[1:], self.kernel_shape)) + tuple(strides[1:])
		a = np.lib.stride_tricks.as_strided(out, shape=shape_, strides=strides)
		a = a.reshape(a.shape[:-len(self.kernel_shape)]+(np.multiply.reduce(a.shape[-len(self.kernel_shape):]),))
		self.masks.append(
			np.unravel_index(np.argmax(a, axis = len(a.shape)-1), self.kernel_shape)
		)

		out = np.amax(a, axis = len(a.shape)-1)
		out = out.reshape(out.shape[0], np.multiply.reduce(out.shape[1:]))

		self.a = self.a + [out]
		return out


	def copy(self):
		return Dropout(self.shape, self.kernel_shape, self.name)

	def __save__dict__(self):
		return {}, [self.shape, self.kernel_shape, self.name]

	def __load__dict__(self, d):
		pass
		

		
