import wbnn
import pickle
import numpy as np

# Define each layer in the Neural Network


layers = [
	wbnn.layers.Input(4096, "Input"),
	wbnn.layers.Convolution((32,128), (5,128), (3,1), 50, "conv1"),
	wbnn.layers.Activation("relu", "ac1"),
	wbnn.layers.Convolution((50,10), (50,3), (1,1), 10, "conv2"),
	wbnn.layers.Activation("relu", "ac2"),
	wbnn.layers.Fully_Connected(4, "logits"),
	wbnn.layers.Activation("softmax", "softmax"),
	wbnn.layers.Loss("ce2", "cross-entropy loss")
]
###################################################################

# Define the connections of the network.
# In this case we have a fully sequential net, i.e. with no parallel
# conections.

for i in range(len(layers)-1):
	layers[i].addNext(layers[i+1])

###################################################################

# Instantiate our Neural Network
nn = wbnn.NN("SimpleASR_conv")

# Tell the net which are its inputs
nn.add_inputs(layers[0])

# Initialize the Neural Network
nn.initialize()


###################################################################

# Load the training/test examples. This needs prepare_Simple_ASR.py to be run.
# In this case everything has already been prepared and "pickled".
# Training data
with open('data/train_spectograms.pickle', 'rb') as handle:
	train_inputs = pickle.load(handle)
with open('data/train_labels.pickle', 'rb') as handle:
	train_outputs = pickle.load(handle)

# Prepare data for convolutional nets.
train_inputs = [x for y in train_inputs for x in y]
train_inputs = [[np.concatenate(train_inputs, axis=1)]]

# Test data
with open('data/test_spectograms.pickle', 'rb') as handle:
	test_inputs = pickle.load(handle)
with open('data/test_labels.pickle', 'rb') as handle:
	test_outputs = pickle.load(handle)

# Prepare data for convolutional nets.
test_inputs = [x for y in test_inputs for x in y]
test_inputs = [[np.concatenate(test_inputs, axis=1)]]


# Initialize the SGD and define its hyperparameters
sgd = wbnn.optimizers.Adam(net=nn, batch_size=128, nb_epochs=40, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

clip = wbnn.optimizers.Clipping(net=nn, min_=-2, max_=2)

sgd.add_regularizer(clip)

# Before training the NN, we define an (optional) function that will be 
# called every epoch. This one will display the training and test loss in each epoch.
# Note that this function needs the epoch, the optimizer, and the training loss
# arguments. The reason of the latter is to have access to the training loss
# without computing it explicitely, since we have already computed and kept it
# at the end of each epoch. It is not exactly as computing it at the end but
# is orientative enough.
def function_for_each_epoch(epoch, optimizer, loss_train):
	loss_dev = optimizer.net.get_loss_of_data((test_inputs, test_outputs))
	print("Epoch: "+str(epoch))
	print("\tTrain loss: "+str(loss_train))
	print("\tTest loss:  "+str(loss_dev))

	# In this case we print the maximum and minimum gradients in the last mini-batch of each epoch
	grads = optimizer.net.get_gradients()
	max_ = 0
	min_ = 0
	for grad in grads:
		new_max = np.max(grad)
		new_min = np.min(grad)
		if new_max > max_:
			max_ = new_max
		if new_min < min_:
			min_ = new_min
	print("max:", max_)
	print("min:", min_)

# Fit the net!
sgd.fit((train_inputs, train_outputs), func_ep=function_for_each_epoch)

# Save the model
nn.save("./models/"+nn.name)
