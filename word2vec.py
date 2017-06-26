import numpy as np
import random
import math

import matplotlib.pyplot as plt

import dnn

		

class DNN():
	def __init__(self, layers):
		self.layers = layers
		self.loss = Loss("ce1")




	def prop(self, input):
		output =  input
		for layer in self.layers:
			output = layer.prop(output)
		return(output)




	def backprop(self, input, desired_output):
		# print(desired_output)
		self.prop(input)
		# print(self.loss.grad(self.rev_layers[0].get_a_during_prop(), desired_output))
		# print(self.loss.ff(self.rev_layers[0].get_a_during_prop(), desired_output))

		# compute the errors
		self.layers[-1].set_loss_error(self.loss.grad(self.layers[-1].get_a_during_prop(), desired_output))
		# where rev_layers[0].get_a_during_prop is just the output of the DNN for the given input.


		for i in range(len(self.layers)-2, -1, -1):
			self.layers[i].backprop_error(self.layers[i+1].get_W(), self.layers[i+1].get_error())

		# compute the gradients
		self.layers[0].compute_gradients(input)

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



if __name__ == '__main__':

	# read the data, Asier's TFG:
	# with open("data_for_w2v/data_all.txt") as f:
	# 	w = f.read().lower().split()

	# print("total words", len(w))
	# print("nb of unique words", len(set(w)))

	# hist = {}
	# for word in w:
	# 	if word in hist:
	# 		hist[word] += 1
	# 	else:
	# 		hist[word] = 1

	# keys = list(hist)
	# keys = sorted(keys, key=lambda name: hist[name],reverse=True)
	# vals = sorted(hist.values(), reverse = True)

	# new_dict = []
	# min_count = 3
	# voc = []
	# for i in range(len(keys)):
	# 	if vals[i] <= min_count:
	# 		break
	# 	new_dict.append([keys[i], vals[i]])
	# 	voc.append(keys[i])



	# print(len(voc))

	# w_subsampled = []

	# for i in range(len(w)):
	# 	if w[i] in voc:
	# 		w_subsampled.append(w[i])

	# with open("data_for_w2v/data_without_min_count_words.txt","w") as f:
	# 	for w in w_subsampled:
	# 		f.write(w+" ")

	###################################################################################

	#load data
	with open("data_for_w2v/data_without_min_count_words.txt") as f:
		w = f.read().lower().split()


	hist = {}
	for word in w:
		if word in hist:
			hist[word] += 1
		else:
			hist[word] = 1

	keys = list(hist)
	keys = sorted(keys, key=lambda name: hist[name],reverse=True)
	vals = sorted(hist.values(), reverse = True)

	total = sum(vals)

	p_word_keep = [x/total for x in vals]
	p_word_keep = [(math.sqrt(x/0.001)+1)*0.001/x for x in p_word_keep]

	to_oneHot = {}
	i = 0
	for word in keys:
		oh = np.zeros(len(keys))
		oh[i] = 1
		to_oneHot[word]= oh
		i+=1

	train_examples = []

	window = 3
	for w_i in range(len(w)):
		for c_i in range(w_i-window, w_i+window):
			if c_i < 0 or c_i >= len(w) or c_i == w_i:
				continue
			if random.random() < p_word_keep[w[w_i]]:
				continue
			train_examples.append((to_oneHot[w[w_i]], to_oneHot[w[c_i]]))
	len(train_examples)


	#plt.plot(sorted(rep))
	#plt.show()


	


	# w2v = DNN([dnn.Fully_Connected_Layer(1000,50, "linear"),
	# 	   dnn.Fully_Connected_Layer(50,2, "softmax")])

