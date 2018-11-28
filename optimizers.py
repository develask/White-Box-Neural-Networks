import numpy as np

class Optimizer():
	def __init__(self, net, batch_size, nb_epochs=None, nb_iterations=None):
		self.net = net
		self.batch_size = batch_size
		self.nb_epochs = nb_epochs
		self.nb_iterations = nb_iterations
		self.regularizers = []

	def compute_gradients(self, m_b):
		return self.net.compute_minibatch_grad(m_b)

	def apply_regularizers(self, grads):
		for regularizer in self.regularizers:
			grads = regularizer.regularize(grads)
		return grads

	def compute_update(self, grads):
		return grads

	def apply_update(self, grads):
		self.net.set_gradients(grads)
		self.net.update_model()

	def train_step(self, m_b):
		grads, loss_m_b = self.compute_gradients(m_b)
		grads = self.apply_regularizers(grads)
		grads = self.compute_update(grads)
		self.apply_update(grads)
		return loss_m_b

	def epoch_loop(self, training_data):
		nb_training_examples = training_data[1][0][0].shape[0]
		indexes = np.arange(nb_training_examples)
		for j in range(self.nb_epochs):
			np.random.shuffle(indexes)
			for i in range(0,nb_training_examples, self.batch_size):
				m_b = ([[input_[indexes[i:i+self.batch_size],:] for input_ in time_step] for time_step in training_data[0]], 
					[[output_[indexes[i:i+self.batch_size],:] for output_ in time_step] for time_step in training_data[1]])
				yield m_b, i+self.batch_size >= nb_training_examples

	def iteration_loop(self, training_data):
		nb_training_examples = training_data[1][0][0].shape[0]
		for i in range(self.nb_iterations):
			indexes = np.random.randint(nb_training_examples, size=self.batch_size)
			m_b = ([[input_[indexes,:] for input_ in time_step] for time_step in training_data[0]], 
					[[output_[indexes,:] for output_ in time_step] for time_step in training_data[1]])
			yield m_b, False

	def iteration_stop_criteria(self, training_data):
		nb_training_examples = training_data[1][0][0].shape[0]
		while self.stop_criteria(self):
			indexes = np.random.randint(nb_training_examples, size=self.batch_size)
			m_b = ([[input_[indexes,:] for input_ in time_step] for time_step in training_data[0]], 
					[[output_[indexes,:] for output_ in time_step] for time_step in training_data[1]])
			yield m_b, False

	def fit(self, training_data,
				func_ep = lambda *x: print(x[0],"training loss:", x[2]),
				func_mb = None):
		if self.nb_iterations and func_mb == None:
			func_mb=func_ep
		self.i_it = self.i_ep = self.loss_ac = 0
		nb_training_examples = training_data[1][0][0].shape[0]
		for m_b, do_func_ep in (self.iteration_loop if self.nb_iterations != None else self.epoch_loop)(training_data):
			loss = self.train_step(m_b)
			self.loss_ac += loss
			if func_mb is not None:
				func_mb(self.i_it, self, loss/m_b[1][0][0].shape[0])
				self.i_it += 1
			if do_func_ep and func_ep is not None:
				func_ep(self.i_ep, self, self.loss_ac/nb_training_examples)
				self.i_ep += 1
				self.loss_ac = 0

	def add_regularizer(self, regularizer):
		assert isinstance(regularizer, Regularizer)
		self.regularizers.append(regularizer)


class SGD(Optimizer):
	def __init__(self, lr_start, lr_end, clipping = None, **kwargs):
		super(SGD, self).__init__(**kwargs)
		self.lr_start = lr_start
		self.lr_end = lr_end
		self.clipping = clipping
		self.lr = self.lr_start
		self.dec_rate = 1

	def compute_update(self, grads):
		grads = [grad*self.lr for grad in grads]
		self.lr *= self.dec_rate
		return grads

	def fit(self, training_data, *args, **kwargs):
		nb_training_examples = training_data[1][0][0].shape[0]
		total_batches = self.nb_iterations if self.nb_iterations != None else self.nb_epochs * np.ceil(nb_training_examples/self.batch_size)
		self.dec_rate = 1
		if total_batches != 1:
			self.dec_rate = (self.lr_end/self.lr_start)**(1/(total_batches-1))
		super(SGD, self).fit(training_data, *args, **kwargs)


class Adam(Optimizer):
	def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
		super(Adam, self).__init__(**kwargs)
		self.alpha = alpha
		self.beta1 = beta1
		self.beta2 = beta2
		self.epsilon = epsilon
		self.net.reset_gradients()
		number_of_gradients = len(self.net.get_gradients())
		self.m = [0]*number_of_gradients 
		self.v = [0]*number_of_gradients 
		self.t = 0
	
	def compute_update(self, grads):
		self.t+=1

		self.m = [self.beta1 * self.m[i] + (1-self.beta1) * grads[i] for i in range(len(grads))]
		self.v = [self.beta2 * self.v[i] + (1-self.beta2) * (grads[i]**2) for i in range(len(grads))]

		m_hat = [m / (1-self.beta1**self.t) for m in self.m]
		v_hat = [v / (1-self.beta2**self.t) for v in self.v]

		grads = [self.alpha * m_hat[i] / (np.sqrt(v_hat[i]) + self.epsilon) for i in range(len(grads))]
		return grads


class Regularizer():
	def __init__(self, net):
		self.net = net

	def regularize(self):
		pass

class Clipping(Regularizer):
	def __init__(self, net, min_, max_):
		super(Clipping, self).__init__(net)
		self.min_ = min_
		self.max_ = max_

	def regularize(self, grads):
		return [np.clip(grad, self.min_, self.max_) for grad in grads]

class L2(Regularizer):
	def __init__(self, net, alpha = 0.0001):
		super(L2, self).__init__(net)
		self.alpha = alpha

	def regularize(self, grads):
		params = self.net.get_params()
		grads = [grads[i] + self.alpha * params[i] for i in range(len(params))]
		return grads

class L1(Regularizer):
	def __init__(self, net, alpha = 0.00001):
		super(L1, self).__init__(net)
		self.alpha = alpha

	def regularize(self, grads):
		params = self.net.get_params()
		grads = [grads[i] + self.alpha * np.sign(params[i]) for i in range(len(params))]
		return grads































