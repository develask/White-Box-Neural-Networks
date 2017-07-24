import dnn
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import time

import layers

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# nn = dnn.DNN("mnist")

# sections = []
# for i in range(16):
# 	nn_ = dnn.DNN("Compute section_"+str(i))
# 	raw = layers.Input(49, "raw_s_"+str(i+1))
# 	h1 = layers.Fully_Connected(64, "sigmoid", "r1_"+str(i+1))
# 	h2 = layers.Fully_Connected(32, "sigmoid", "r2_"+str(i+1))

# 	raw.addNext(h1)
# 	h1.addNext(h2)

# 	# h1.addNextRecurrent(h1)
# 	# h2.addNextRecurrent(h2)

# 	nn_.add_inputs(raw)

# 	sections.append(nn_)


# raws = [layers.Input(49, "raw "+str(i+1)) for i in range(16)]

# h1s = [layers.Fully_Connected(28, "sigmoid", "h1_"+str(i+1)) for i in range(16)]
# h2s = [layers.Fully_Connected(16, "sigmoid", "h2_"+str(i+1)) for i in range(16)]

# h3 = layers.Fully_Connected(128, "sigmoid", "h3")

# ou = layers.Fully_Connected(10, "softmax", "output")

# for i in range(16):
# 	raws[i].addNext(sections[i])
	# sections[i].addNext(h3)

# for i in range(16):
# 	raws[i].addNext(h1s[i])
# 	h1s[i].addNext(h2s[i])
# 	h2s[i].addNext(h3)
# h3.addNext(ou)

# for i in range(16):
# 	nn.add_inputs(raws[i])

# nn.initialize()


# print("data transposing")

# xs, ys = mnist.train.next_batch(mnist.train.num_examples)
# training_data_tmp = list(zip(xs,ys))
# training_data = []

def convert_img(img):
	im = img.reshape((28,28))
	inputs = []
	for i in range(4):
		x_min = i*7
		x_max = x_min+7
		for j in range(4):
			y_min = j*7
			y_max = y_min + 7
			inputs.append(im[x_min:x_max,y_min:y_max])
	return [[m.reshape((49,1)) for m in inputs]]
	# for i in range(inputs[0].shape[0]):
	# 	tmp = []
	# 	for box in inputs:
	# 		tmp.append(box[i,:,np.newaxis])
	# 	inp.append(tmp)
	# return inp
	#return [ img[:,np.newaxis] ]


# for example in training_data_tmp[:200]:
# 	training_data.append((convert_img(example[0]), example[1][:,np.newaxis]))

# len_data = len(training_data)

# print("data training")
# start = time.clock()
# nn.SGD(training_data[int(len_data*0.01):], training_data[:int(len_data*0.01)], 128, 15, 0.5, 0.1)
# end = time.clock()
# interval_c = end-start
# print("Time training: %.2gs" % interval_c)

# nn.save()

nn = dnn.DNN.load('mnist_2017-07-24_16:46')

print("data testing")
xs, ys = mnist.test.next_batch(mnist.test.num_examples)
good = 0
total = 0
rmat = np.zeros((10,10))
fallados = []
for i in range(mnist.test.num_examples):
	em = nn.prop(convert_img(xs[i]))
	pred = np.argmax(em)
	real = np.argmax(ys[i])
	rmat[pred,real] += 1
	if pred == real:
		good+=1
	else:
		fallados.append([xs[i].reshape((28,28)), pred, real])
	total+=1
print("rate:", good/total, "(",good,"/",total,")")
print("-----------------------------------------------")
print("row is real")
print("col is predicted")
print(rmat.astype(int))
q = input("> quieres ver las fotos que falla ("+str(len(fallados))+") (s/n)?  ")
if q == "s":
	import matplotlib.pyplot as plt
	for fallo in fallados:
		plt.title("pred: "+str(fallo[1])+' - real: '+str(fallo[2]))
		plt.imshow(fallo[0])
		plt.show()
