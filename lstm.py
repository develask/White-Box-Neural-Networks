from dnn import Layer
import numpy as np

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
		return np.dot(np.transpose(self.W_o_alprevs[idx]), self.get_o_error(t=t) * self.__dev_sigm__(self.get_o(t=t)))
		+ np.dot(np.transpose(self.W_c_alprevs[idx]), self.get_c_error(t=t) * self.get_i(t=t) * self.__dev_tanh__(self.get_c(t=t)))
		+ np.dot(np.transpose(self.W_i_alprevs[idx]), self.get_i_error(t=t) * self.__dev_sigm__(self.get_i(t=t)))
		+ np.dot(np.transpose(self.W_f_alprevs[idx]), self.get_f_error(t=t) * self.__dev_sigm__(self.get_f(t=t)))

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
				np.transpose(self.W_o_atprev),
				self.get_o_error(t=t+1) * self.__dev_sigm__(self.get_o(t=t+1))
			) + \
			np.dot(
				np.transpose(self.W_c_atprev),
				self.get_c_error(t=t+1) * self.get_i(t=t+1) * self.__dev_tanh__(self.get_c(t=t+1))
			) + \
			np.dot(
				np.transpose(self.W_i_atprev),
				self.get_i_error(t=t+1) * self.__dev_sigm__(self.get_i(t=t+1))
			) + \
			np.dot(
				np.transpose(self.W_f_atprev),
				self.get_f_error(t=t+1) * self.__dev_sigm__(self.get_f(t=t+1))
			)
		self.error_a = [aux]+self.error_a

		# error in o
		self.error_o = [self.get_a_error(t=t) * self.__tanh__(self.get_c(t=t))] + self.error_o

		# error in c
		aux = self.w_o_c * self.get_o_error(t=t) * self.__dev_sigm__(self.get_o(t=t))
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
					[np.dot(aux_o, np.transpose(l.get_Output(t=t-1))) for l in self.prev_recurrent]+\
					[np.dot(aux_o, np.transpose(l.get_Output(t=t))) for l in self.prev]
				))

				W_c_alprevs_grads_aux = list(map(lambda x,y: x+y,
					W_c_alprevs_grads_aux,
					[np.dot(aux_c, np.transpose(l.get_Output(t=t-1))) for l in self.prev_recurrent]+\
					[np.dot(aux_c, np.transpose(l.get_Output(t=t))) for l in self.prev]
				))

				W_f_alprevs_grads_aux = list(map(lambda x,y: x+y,
					W_f_alprevs_grads_aux,
					[np.dot(aux_f, np.transpose(l.get_Output(t=t-1))) for l in self.prev_recurrent]+\
					[np.dot(aux_f, np.transpose(l.get_Output(t=t))) for l in self.prev]
				))

				W_i_alprevs_grads_aux = list(map(lambda x,y: x+y,
					W_i_alprevs_grads_aux,
					[np.dot(aux_i, np.transpose(l.get_Output(t=t-1))) for l in self.prev_recurrent]+\
					[np.dot(aux_i, np.transpose(l.get_Output(t=t))) for l in self.prev]
				))

			W_i_atprev_grads_aux += np.dot(aux_i, np.transpose(self.get_Output(t=t-1)))
			W_f_atprev_grads_aux += np.dot(aux_f, np.transpose(self.get_Output(t=t-1)))
			W_c_atprev_grads_aux += np.dot(aux_c, np.transpose(self.get_Output(t=t-1)))
			W_o_atprev_grads_aux += np.dot(aux_o, np.transpose(self.get_Output(t=t-1)))

			w_i_ctprev_grads_aux += aux_i * self.get_c(t=t-1)
			w_f_ctprev_grads_aux += aux_f * self.get_c(t=t-1)
			w_o_c_grads_aux += aux_o * self.get_c(t=t)

			b_i_grads_aux = aux_i
			b_f_grads_aux = aux_f
			b_c_grads_aux = aux_c
			b_o_grads_aux = aux_o

		self.apply_to_gradients(lambda x: x/num_t)

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

		self.W_i_alprevs = [np.random.normal(scale=1 ,size=(self.get_Output_dim(), l.get_Output_dim())) for l in self.layers]
		self.W_f_alprevs = [np.random.normal(scale=1 ,size=(self.get_Output_dim(), l.get_Output_dim())) for l in self.layers]
		self.W_c_alprevs = [np.random.normal(scale=1 ,size=(self.get_Output_dim(), l.get_Output_dim())) for l in self.layers]
		self.W_o_alprevs = [np.random.normal(scale=1 ,size=(self.get_Output_dim(), l.get_Output_dim())) for l in self.layers]

		self.W_i_atprev = np.random.normal(scale=1 ,size=(self.get_Output_dim(), self.get_Output_dim()))
		self.W_f_atprev = np.random.normal(scale=1 ,size=(self.get_Output_dim(), self.get_Output_dim()))
		self.W_c_atprev = np.random.normal(scale=1 ,size=(self.get_Output_dim(), self.get_Output_dim()))
		self.W_o_atprev = np.random.normal(scale=1 ,size=(self.get_Output_dim(), self.get_Output_dim()))

		self.w_i_ctprev = np.random.normal(scale=1 ,size=(self.get_Output_dim(), 1))
		self.w_f_ctprev = np.random.normal(scale=1 ,size=(self.get_Output_dim(), 1))
		self.w_o_c = np.random.normal(scale=1 ,size=(self.get_Output_dim(), 1))

		self.b_i = np.random.normal(scale=1 ,size=(self.get_Output_dim(), 1))
		self.b_f = np.random.normal(scale=1 ,size=(self.get_Output_dim(), 1))
		self.b_c = np.random.normal(scale=1 ,size=(self.get_Output_dim(), 1))
		self.b_o = np.random.normal(scale=1 ,size=(self.get_Output_dim(), 1))


	def prop(self):
		len_prev_rec = len(self.prev_recurrent)

		self.i += [None]
		self.f += [None]
		self.c += [None]
		self.o += [None]
		self.a += [None]

		self.i[-1] = self.__sigm__(sum([
				np.dot(self.W_i_alprevs[l_i], self.prev_recurrent[l_i].get_Output(t=-1))
				for l_i in range(len(self.prev_recurrent))
			])\
			+ sum([
				np.dot(self.W_i_alprevs[len_prev_rec+l_i], self.prev[l_i].get_Output(t=0))
					for l_i in range(len(self.prev))
			])\
			+ np.dot(self.W_i_atprev, self.get_Output(t=-1))\
			+ self.w_i_ctprev * self.get_c(t=-1)\
			+ self.b_i
		)

		self.f[-1] = self.__sigm__(sum([
				np.dot(self.W_f_alprevs[l_i], self.prev_recurrent[l_i].get_Output(t=-1))
				for l_i in range(len(self.prev_recurrent))
			])\
			+ sum([
				np.dot(self.W_f_alprevs[len_prev_rec+l_i], self.prev[l_i].get_Output(t=0))
					for l_i in range(len(self.prev))
			])\
			+ np.dot(self.W_f_atprev, self.get_Output(t=-1))\
			+ self.w_f_ctprev * self.get_c(t=-1)\
			+ self.b_f
		)

		self.c[-1] = self.get_f(t=0) * self.get_c(t=-1) \
			+ self.__tanh__(
				sum([
					np.dot(self.W_c_alprevs[l_i], self.prev_recurrent[l_i].get_Output(t=-1))
					for l_i in range(len(self.prev_recurrent))
				])\
				+ sum([
					np.dot(self.W_c_alprevs[len_prev_rec+l_i], self.prev[l_i].get_Output(t=0))
						for l_i in range(len(self.prev))
				])\
				+ np.dot(self.W_c_atprev, self.get_Output(t=-1))\
				+ self.b_c
			)
		self.o[-1] = self.__sigm__(sum([
				np.dot(self.W_o_alprevs[l_i], self.prev_recurrent[l_i].get_Output(t=-1))
				for l_i in range(len(self.prev_recurrent))
			])\
			+ sum([
				np.dot(self.W_o_alprevs[len_prev_rec+l_i], self.prev[l_i].get_Output(t=0))
					for l_i in range(len(self.prev))
			])\
			+ np.dot(self.W_o_atprev, self.get_Output(t=-1))\
			+ self.w_o_c * self.get_c(t=0)\
			+ self.b_o
		)

		self.a[-1] = self.get_o(t=0) * self.__tanh__(self.get_c(t=0))

	def reset(self):
		self.a = [np.zeros((self.num_cel,1))]
		self.i = []
		self.f = []
		self.c = [np.zeros((self.num_cel,1))]
		self.o = []
		self.error_a = []
		self.error_o = []
		self.error_c = []
		self.error_f = []
		self.error_i = []

if __name__ == '__main__':
	import random
	import math
	import time

	import dnn
	
	nn = dnn.DNN("Our LSTM")

	x = dnn.Input(1, "x")

	def func(x):
		return math.sin(sum([a[0] for a in x]))**2

	R1 = LSTM(100, "r1")
	R2 = LSTM(100, "r2")
	R3 = LSTM(100, "r3")

	ff = dnn.Fully_Connected(1, "sigmoid", "ff2")


	x.addNext(R1)
	R1.addNext(R2)
	R2.addNext(R3)
	R3.addNext(ff)

	nn.add_inputs(x)

	nn.initialize()
	training_data = []


	for i in range(20000):
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

	nn.SGD(training_data[500:], training_data[:500], 128, 10, 0.5, 0.5)

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