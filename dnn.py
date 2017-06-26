import numpy as np
import random
import math


class Activation_Function():
	def __init__(self, name):
		self.name = name

	def ff(self, z):
		if self.name == "sigmoid":
			return 1 / (1 + np.exp(-z))

		elif self.name == "softmax":
			exp = np.exp(z)
			return exp/np.sum(exp)

		elif self.name == "relu":
			return np.maximum(z, 0)

		elif self.name == "linear":
			return z

		else:
			raise ValueError("Not defined activation function")

	def derivative(self, z):
		if self.name == "sigmoid":
			return np.exp(-z)/(1+np.exp(-z))**2

		elif self.name == "softmax":
			exp = np.exp(z)
			suma = np.sum(exp)
			div = exp / suma
			return div - div**2

		elif self.name == "relu":
			return (np.sign(z)+1)/2

		elif self.name == "linear":
			return np.full(z.shape, 1)

		else:
			raise ValueError("Not defined activation function")

	def derivative_ff(self, a):
		# compute the derivative in function of the ff function, the activation
		if self.name == "sigmoid":
			return a*(1-a)

		elif self.name == "softmax":
			return a - a**2

		elif self.name == "relu":
			return np.sign(a)

		elif self.name == "linear":
			return a/a
			return np.full(a.shape, 1)

		else:
			raise ValueError("Not defined activation function")


class Fully_Connected_Layer():
	def __init__(self, input_dim, output_dim, activation_function):
		'''
		input_dim: input dimension of the layer
		output_dim: output dimension of the layer
		activation_function: string to select an activation function, e.g. "sigmoid"
		'''

		'''
		W is a output_dim x input_dim matrix, so w_ij corresponds to the weight
		between the j_th neuron in the input to the i_th neuron in the output
		b is the bias vector
		act_f refers to the selected activation function

		z is the input to the activation during the propagation stage
		a is the output of the activation during the propagation stage

		error is the backprop error in this layer
		'''

		self.W = np.random.normal(scale=1 ,size=(output_dim, input_dim))
		#self.W = np.zeros((output_dim, input_dim))
		self.b = np.zeros((output_dim, 1))
		self.act_f = Activation_Function(activation_function)

		self.z = np.zeros((output_dim,1))
		self.a = np.zeros((output_dim,1))

		self.error = np.zeros((output_dim,1))

		self.b_grad = np.zeros((output_dim,1))
		self.W_grad = np.zeros((output_dim,input_dim))

	def prop(self, inp):
		'''
		inp is assumed to be of shape (inp_shape, 1)
		'''
		self.z = np.dot(self.W, inp) + self.b
		self.a = self.act_f.ff(self.z)
		return self.a

	def get_b(self):
		return self.b

	def get_W(self):
		return self.W

	def get_z_during_prop(self):
		return self.z

	def get_a_during_prop(self):
		return self.a

	def get_error(self):
		return self.error

	def backprop_error(self, W_next_layer, error_next_layer):
		self.error = np.dot(np.transpose(W_next_layer), error_next_layer) * self.act_f.derivative_ff(self.a)


	def set_loss_error(self, loss_gradient):
		# this function will be used in the first step of the BP., when the error is set from 
		# the cost function (in supervised learning)
		self.error = loss_gradient * self.act_f.derivative_ff(self.a)
		#self.error = loss_gradient * self.act_f.derivative(self.z)

	def compute_gradients(self, inp):
		#inp is a^{l-1}
		self.b_grad = self.error
		self.W_grad = np.dot(self.error, np.transpose(inp))

	def get_W_gradient(self):
		return self.W_grad

	def get_b_gradient(self):
		return self.b_grad

	def update_W(self, increment):
		self.W += increment

	def update_b(self, increment):
		self.b += increment

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
	def __init__(self, layers):
		self.layers = layers
		self.loss = Loss("ce1")




	def prop(self, inp):
		output =  inp
		for layer in self.layers:
			output = layer.prop(output)
		return(output)




	def backprop(self, inp, desired_output):
		# print(desired_output)
		self.prop(inp)
		# print(self.loss.grad(self.rev_layers[0].get_a_during_prop(), desired_output))
		# print(self.loss.ff(self.rev_layers[0].get_a_during_prop(), desired_output))

		# compute the errors
		self.layers[-1].set_loss_error(self.loss.grad(self.layers[-1].get_a_during_prop(), desired_output))
		# where rev_layers[0].get_a_during_prop is just the output of the DNN for the given inp.


		for i in range(len(self.layers)-2, -1, -1):
			self.layers[i].backprop_error(self.layers[i+1].get_W(), self.layers[i+1].get_error())

		# compute the gradients
		self.layers[0].compute_gradients(inp)

		for i in range(1, len(self.layers)):
			self.layers[i].compute_gradients(self.layers[i-1].get_a_during_prop())




	def SGD(self, training_data, batch_size, nb_epochs, lr_start, lr_end):
		lr = lr_start
		dec_rate = (lr_end/lr_start)**(1/(nb_epochs-1))
		W_grads = []
		b_grads = []
		len_tr = len(training_data)

		for i in range(nb_epochs):
			l = 0
			out_1 = 0
			for ex in training_data:
				out = self.prop(ex[0])
				l += self.loss.ff(out,ex[1])/len_tr
				out_1 += out[0][0]/len_tr

			print("training loss", l[0])

			random.shuffle(training_data)
			mini_batches = [ training_data[k:k+batch_size] for k in range(0, len_tr, batch_size)]
			
			for m_b in mini_batches:
				# update the DNN according to this mini batch

				# compute the errors and gradients per example
				len_m_b = len(m_b)
				W_grads = []
				b_grads = []

				self.backprop(*m_b[0])
				for k in range(len(self.layers)):
					W_grads.append(self.layers[k].get_W_gradient()*lr/len_m_b)
					b_grads.append(self.layers[k].get_b_gradient()*lr/len_m_b)

				# for m in range(len(self.layers)):
				# 	print("Weight gradient matrix crontibution nb of layer", m, "in example 0 of the minibatch")
				# 	print(-W_grads[m])
				# print("----------------------------------------------")


				for j in range(1,len(m_b)):
					self.backprop(*m_b[j])
					for k in range(len(self.layers)):
						W_grads[k] += self.layers[k].get_W_gradient()*lr/len_m_b
						b_grads[k] += self.layers[k].get_b_gradient()*lr/len_m_b

				# update weights and biases:
				for k in range(len(self.layers)):
					self.layers[k].update_b(-b_grads[k])
					self.layers[k].update_W(-W_grads[k])

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

	dnn = DNN([Fully_Connected_Layer(1,50, "sigmoid"),
			   Fully_Connected_Layer(50,50, "sigmoid"),
			   Fully_Connected_Layer(50,50, "sigmoid"),
			   Fully_Connected_Layer(50,50, "sigmoid"),
			   Fully_Connected_Layer(50,2, "softmax")])


	### generate some data...

	training_data = []

	for i in range(20000):
		x = random.random()
		training_data.append((np.asarray([[x]]), np.asarray([[ math.sin(x)**2 ], [ math.cos(x)**2 ]] )))

	print("gonna train, holly shit I'm nervous")
	dnn.SGD(training_data, 1, 10, 0.005, 0.0005)

	print("let's compute x = 0.3")
	print(dnn.prop(np.asarray([[0.3]])))

	print("let's compute x = 0.51")
	print(dnn.prop(np.asarray([[0.51]])))

	print("let's compute x = 0.9")
	print(dnn.prop(np.asarray([[0.9]])))






