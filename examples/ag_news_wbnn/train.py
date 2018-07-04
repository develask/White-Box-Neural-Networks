import numpy as np
import wbnn
from lookuptable import LookUpTable

# Format the inputs to feed them into the net...
def convert_inputs(inps):
	return [ [ inps[:,i][:,np.newaxis] ] for i in range(inps.shape[1])]


# Load the inputs and outputs
train_inputs = np.load("data/train_inputs.npy").astype(np.int32)
train_class = np.load("data/train_labels.npy")

test_inputs = np.load("data/test_inputs.npy").astype(np.int32)
test_class = np.load("data/test_labels.npy")

train_inputs = convert_inputs(train_inputs)
test_inputs = convert_inputs(test_inputs)

# Load the train/test embeddings. This will be used by the LookUp layer.
# The dimension has been set to 50, but it can be changed if other lookups are prepared
# during the prepare_ag_news.py script
embeddings_train = np.load("data/train_embeddings_50.npy")
embeddings_test = np.load("data/test_embeddings_50.npy")



# Define the layers of the NN
x = wbnn.layers.Input(1, "Input")
lu = wbnn.layers.LookUpTable(embeddings_train, "lookup") # New layer!!

lstm = wbnn.layers.LSTM(15, "lstm")

fc = wbnn.layers.Fully_Connected(4, "logits")
sm = wbnn.layers.Activation("softmax", "softmax")

ce_loss = wbnn.layers.Loss("ce2", "cross-entropy loss")

# Change the initializer of the LSTMs, these usually need low starting values.
init = wbnn.layers.initializers.RandomNormal(mean=0, stddev=0.001)
lstm.setInitializer(init)

# Define the architecture of the net.
x.addNext(lu)
lu.addNext(lstm)
lstm.addNext(fc)
fc.addNext(sm)
sm.addNext(ce_loss)

# Instantiate the NN
nn = wbnn.NN("LSTM_baseline")

# Tell the net which are its inputs
nn.add_inputs(x)

# Initialize the Neural Network
nn.initialize()


sgd = wbnn.optimizers.Adam(net=nn, batch_size=256, nb_iterations=1000, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

clip = wbnn.optimizers.Clipping(net=nn, min_=-2, max_=2)
sgd.add_regularizer(clip)

# In this case we will print the test/train losses but also save the model in each epoch.
def function_for_each_epoch(epoch, optimizer, loss_train):
	if epoch % 10 == 9:
		# Set the test matrix for the test loss
		lu.lookup = embeddings_test
		loss_dev = optimizer.net.get_loss_of_data((test_inputs, [[test_class]]))
		# Then put the train embeddings back
		lu.lookup = embeddings_train
		print("Epoch: "+str(epoch//10))
		print("\tTrain loss: "+str(loss_train))
		print("\tTest loss:  "+str(loss_dev))
		nn.save("./models/"+nn.name+"_it"+str(epoch//10))


# Fit the net!
sgd.fit((train_inputs, [[train_class]]), func_ep=function_for_each_epoch)

