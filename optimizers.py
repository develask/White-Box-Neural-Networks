import numpy as np

class Optimizer():
	def __init__(self, net):
		self.net = net


class SGD(Optimizer):
	def __init__(self, net, batch_size, nb_epochs, lr_start, lr_end, clipping = None):
		super(SGD, self).__init__(net)
		self.batch_size = batch_size
		self.nb_epochs = nb_epochs
		self.lr_start = lr_start
		self.lr_end = lr_end
		self.clipping = clipping
	
	def train_step(self, m_b):
		grads, loss_m_b = self.net.compute_minibatch_grad(m_b)
		if self.clipping is not None:
			self.net.apply_to_gradients(lambda grad: np.clip(grad, *self.clipping))
		self.net.apply_to_gradients(lambda x: x*self.lr)
		self.net.update_model()
		return loss_m_b

	def fit(self, training_data,
				func = lambda *x: print(x[0],"training loss:", x[2])):
		self.lr = self.lr_start
		dec_rate = 1
		if self.nb_epochs != 1:
			dec_rate = (self.lr_end/self.lr_start)**(1/(self.nb_epochs-1))

		nb_training_examples = training_data[1][0][0].shape[0]
		indexes = np.arange(nb_training_examples)

		for j in range(self.nb_epochs):
			loss = 0
			np.random.shuffle(indexes)
			for i in range(0,nb_training_examples, self.batch_size):
				m_b = ([[input_[indexes[i:i+self.batch_size],:] for input_ in time_step] for time_step in training_data[0]], 
					[[output_[indexes[i:i+self.batch_size],:] for output_ in time_step] for time_step in training_data[1]])
				loss += self.train_step(m_b)

			if func is not None:
				func(j, self, loss/nb_training_examples)
			self.lr *= dec_rate
