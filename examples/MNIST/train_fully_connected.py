import wbnn
import numpy as np

# Define each layer in the Neural Network

# Input layer
x = wbnn.layers.Input(784, "Input") # 28 x 28 = 784

# Hidden layers
#  fully connected layer
fc1 = wbnn.layers.Fully_Connected(250, "fc1")
#  its activation function
ac1 = wbnn.layers.Activation("sigmoid", "ac1")

# second hidden layer
fc2 = wbnn.layers.Fully_Connected(150, "fc2")
ac2 = wbnn.layers.Activation("sigmoid", "ac2")

# output layer: a softmax of 10 classes
fc3 = wbnn.layers.Fully_Connected(10, "logits")
sm = wbnn.layers.Activation("softmax", "softmax")

# Cross entropy loss
ce_loss = wbnn.layers.Loss("ce1", "cross-entropy loss")

###################################################################

# Define the connections of the network.
# In this case we have a fully sequential net, i.e. with no parallel
# conections.

x.addNext(fc1)
fc1.addNext(ac1)
ac1.addNext(fc2)
fc2.addNext(ac2)
ac2.addNext(fc3)
fc3.addNext(sm)
sm.addNext(ce_loss)

###################################################################

# Instantiate our Neural Network
nn = wbnn.NN("MNIST_fully_connected")

# Tell the net which are its inputs
nn.add_inputs(x)

# Initialize the Neural Network
nn.initialize()


###################################################################

# Load the training/test examples. This needs prepare_MNIST to be run.

# Training data
train_inputs = [[np.load("./data/train_images.npy")]]
train_outputs = [[np.load("./data/train_labels.npy")]]

# Test data
test_inputs = [[np.load("./data/t10k_images.npy")]]
test_outputs = [[np.load("./data/t10k_labels.npy")]]

# Initialize the SGD and define its hyperparameters
sgd = wbnn.optimizers.SGD(nn, batch_size=128, nb_epochs=50, lr_start=0.5, lr_end=0.2)


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

# Fit the net!
sgd.fit((train_inputs, train_outputs), func=function_for_each_epoch)

# Save the model
nn.save("./models/"+nn.name)
