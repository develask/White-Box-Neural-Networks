import wbnn
import numpy as np

# Define each layer in the Neural Network. Since the network will be
# fully sequential, we can define all the layers in a list to define
# the structure easier.

layers = [
        wbnn.layers.Input(784, "Input"),

        wbnn.layers.Convolution((28,28), (5,5), (1,1), 5, "conv1"),
        wbnn.layers.MaxPooling((5,24,24), (1,2,2), "pool1"),
        wbnn.layers.Activation("sigmoid", "ac1"),
        wbnn.layers.Dropout(0.2, "drp1"),

        wbnn.layers.Convolution((5,12,12), (5,3,3), (1,1,1), 10, "conv2"),
        wbnn.layers.MaxPooling((10,10,10), (1,2,2), "pool2"),
        wbnn.layers.Activation("sigmoid", "ac2"),
        wbnn.layers.Dropout(0.2, "drp2"),

        wbnn.layers.Fully_Connected(10, 'fc2'),
        wbnn.layers.Activation("softmax", "ac4"),

        wbnn.layers.Loss('ce1', 'loss')
]

###################################################################

# Define the connections of the network.
# In this case we also have a fully sequential net, i.e. with no parallel
# conections. Hence we can make the connections with this for loop.

for i in range(len(layers)-1):
        layers[i].addNext(layers[i+1])

###################################################################

# Instantiate our Neural Network
nn = wbnn.NN("MNIST_convol_maxpool")

# Tell the net which are its inputs
nn.add_inputs(layers[0])

# Initialize the Neural Network
nn.initialize()


###################################################################

# Load the training/test examples. This needs prepare_MNIST.py to be run.

# Training data
# The numpy arrays we have saved in   
train_inputs = [[np.load("./data/train_images.npy")]]
train_outputs = [[np.load("./data/train_labels.npy")]]

# Test data
test_inputs = [[np.load("./data/t10k_images.npy")]]
test_outputs = [[np.load("./data/t10k_labels.npy")]]

# Initialize the SGD and define its hyperparameters
sgd = wbnn.optimizers.SGD(nn, batch_size=128, nb_epochs=50, lr_start=1, lr_end=1)

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
	# In this case we will make partial saves each 10 epochs
	if epoch % 10 == 0 and epoch != 0:
		optimizer.net.save("./models/"+nn.name+"_ep"+str(epoch))
        print("Saved model:", "./models/"+nn.name+"_ep"+str(epoch))

# Fit the net! This might take around 40-60 minutes
sgd.fit((train_inputs, train_outputs), func=function_for_each_epoch)

# Save the model
nn.save("./models/"+nn.name)