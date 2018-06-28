import wbnn
from wbnn import logger
import numpy as np

logs = logger.Logger('logs')

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
test_inputs = np.load("./data/t10k_images.npy")
test_outputs = np.load("./data/t10k_labels.npy")

indexes = np.concatenate([np.where(test_outputs[:,i] == 1)[0] for i in range(10)],axis=0)

test_inputs = [[test_inputs[indexes,:]]]
test_outputs = [[test_outputs[indexes,:]]]

# Initialize the SGD and define its hyperparameters
sgd = wbnn.optimizers.SGD(nn, batch_size=128, nb_epochs=50, lr_start=1, lr_end=0.5)

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
	logs.scalar('dev_loss', loss_dev)
	idx = int(np.random.rand() * 1000)
	logs.image('image 0', layers[0].get_Output()[0000+idx,:].reshape(28,28))
	logs.image('image 1', layers[0].get_Output()[1000+idx,:].reshape(28,28))
	logs.image('image 2', layers[0].get_Output()[2000+idx,:].reshape(28,28))
	logs.image('image 3', layers[0].get_Output()[3000+idx,:].reshape(28,28))
	logs.image('image 4', layers[0].get_Output()[4000+idx,:].reshape(28,28))
	logs.image('image 5', layers[0].get_Output()[5000+idx,:].reshape(28,28))
	logs.image('image 6', layers[0].get_Output()[6000+idx,:].reshape(28,28))
	logs.image('image 7', layers[0].get_Output()[7000+idx,:].reshape(28,28))
	logs.image('image 8', layers[0].get_Output()[8000+idx,:].reshape(28,28))
	logs.image('image 9', layers[0].get_Output()[9000+idx,:].reshape(28,28))

	logs.image('conv1_0 f', layers[1].W[0])
	
	logs.image('conv1_0 0', layers[3].get_Output()[0000+idx,:].reshape(5,12,12)[0,...])
	logs.image('conv1_0 1', layers[3].get_Output()[1000+idx,:].reshape(5,12,12)[0,...])
	logs.image('conv1_0 2', layers[3].get_Output()[2000+idx,:].reshape(5,12,12)[0,...])
	logs.image('conv1_0 3', layers[3].get_Output()[3000+idx,:].reshape(5,12,12)[0,...])
	logs.image('conv1_0 4', layers[3].get_Output()[4000+idx,:].reshape(5,12,12)[0,...])
	logs.image('conv1_0 5', layers[3].get_Output()[5000+idx,:].reshape(5,12,12)[0,...])
	logs.image('conv1_0 6', layers[3].get_Output()[6000+idx,:].reshape(5,12,12)[0,...])
	logs.image('conv1_0 7', layers[3].get_Output()[7000+idx,:].reshape(5,12,12)[0,...])
	logs.image('conv1_0 8', layers[3].get_Output()[8000+idx,:].reshape(5,12,12)[0,...])
	logs.image('conv1_0 9', layers[3].get_Output()[9000+idx,:].reshape(5,12,12)[0,...])
	
	logs.image('conv1_1 f', layers[1].W[1])

	logs.image('conv1_1 0', layers[3].get_Output()[0000+idx,:].reshape(5,12,12)[1,...])
	logs.image('conv1_1 1', layers[3].get_Output()[1000+idx,:].reshape(5,12,12)[1,...])
	logs.image('conv1_1 2', layers[3].get_Output()[2000+idx,:].reshape(5,12,12)[1,...])
	logs.image('conv1_1 3', layers[3].get_Output()[3000+idx,:].reshape(5,12,12)[1,...])
	logs.image('conv1_1 4', layers[3].get_Output()[4000+idx,:].reshape(5,12,12)[1,...])
	logs.image('conv1_1 5', layers[3].get_Output()[5000+idx,:].reshape(5,12,12)[1,...])
	logs.image('conv1_1 6', layers[3].get_Output()[6000+idx,:].reshape(5,12,12)[1,...])
	logs.image('conv1_1 7', layers[3].get_Output()[7000+idx,:].reshape(5,12,12)[1,...])
	logs.image('conv1_1 8', layers[3].get_Output()[8000+idx,:].reshape(5,12,12)[1,...])
	logs.image('conv1_1 9', layers[3].get_Output()[9000+idx,:].reshape(5,12,12)[1,...])
	
	logs.image('conv1_2 f', layers[1].W[2])

	logs.image('conv1_2 0', layers[3].get_Output()[0000+idx,:].reshape(5,12,12)[2,...])
	logs.image('conv1_2 1', layers[3].get_Output()[1000+idx,:].reshape(5,12,12)[2,...])
	logs.image('conv1_2 2', layers[3].get_Output()[2000+idx,:].reshape(5,12,12)[2,...])
	logs.image('conv1_2 3', layers[3].get_Output()[3000+idx,:].reshape(5,12,12)[2,...])
	logs.image('conv1_2 4', layers[3].get_Output()[4000+idx,:].reshape(5,12,12)[2,...])
	logs.image('conv1_2 5', layers[3].get_Output()[5000+idx,:].reshape(5,12,12)[2,...])
	logs.image('conv1_2 6', layers[3].get_Output()[6000+idx,:].reshape(5,12,12)[2,...])
	logs.image('conv1_2 7', layers[3].get_Output()[7000+idx,:].reshape(5,12,12)[2,...])
	logs.image('conv1_2 8', layers[3].get_Output()[8000+idx,:].reshape(5,12,12)[2,...])
	logs.image('conv1_2 9', layers[3].get_Output()[9000+idx,:].reshape(5,12,12)[2,...])
	
	logs.image('conv1_3 f', layers[1].W[3])

	logs.image('conv1_3 0', layers[3].get_Output()[0000+idx,:].reshape(5,12,12)[3,...])
	logs.image('conv1_3 1', layers[3].get_Output()[1000+idx,:].reshape(5,12,12)[3,...])
	logs.image('conv1_3 2', layers[3].get_Output()[2000+idx,:].reshape(5,12,12)[3,...])
	logs.image('conv1_3 3', layers[3].get_Output()[3000+idx,:].reshape(5,12,12)[3,...])
	logs.image('conv1_3 4', layers[3].get_Output()[4000+idx,:].reshape(5,12,12)[3,...])
	logs.image('conv1_3 5', layers[3].get_Output()[5000+idx,:].reshape(5,12,12)[3,...])
	logs.image('conv1_3 6', layers[3].get_Output()[6000+idx,:].reshape(5,12,12)[3,...])
	logs.image('conv1_3 7', layers[3].get_Output()[7000+idx,:].reshape(5,12,12)[3,...])
	logs.image('conv1_3 8', layers[3].get_Output()[8000+idx,:].reshape(5,12,12)[3,...])
	logs.image('conv1_3 9', layers[3].get_Output()[9000+idx,:].reshape(5,12,12)[3,...])
	
	logs.image('conv1_4 f', layers[1].W[4])

	logs.image('conv1_4 0', layers[3].get_Output()[0000+idx,:].reshape(5,12,12)[4,...])
	logs.image('conv1_4 1', layers[3].get_Output()[1000+idx,:].reshape(5,12,12)[4,...])
	logs.image('conv1_4 2', layers[3].get_Output()[2000+idx,:].reshape(5,12,12)[4,...])
	logs.image('conv1_4 3', layers[3].get_Output()[3000+idx,:].reshape(5,12,12)[4,...])
	logs.image('conv1_4 4', layers[3].get_Output()[4000+idx,:].reshape(5,12,12)[4,...])
	logs.image('conv1_4 5', layers[3].get_Output()[5000+idx,:].reshape(5,12,12)[4,...])
	logs.image('conv1_4 6', layers[3].get_Output()[6000+idx,:].reshape(5,12,12)[4,...])
	logs.image('conv1_4 7', layers[3].get_Output()[7000+idx,:].reshape(5,12,12)[4,...])
	logs.image('conv1_4 8', layers[3].get_Output()[8000+idx,:].reshape(5,12,12)[4,...])
	logs.image('conv1_4 9', layers[3].get_Output()[9000+idx,:].reshape(5,12,12)[4,...])
	
	logs.image('conv2_0 f', layers[5].W[0])
	logs.image('conv2_1 f', layers[5].W[1])
	logs.image('conv2_2 f', layers[5].W[2])
	logs.image('conv2_3 f', layers[5].W[3])
	logs.image('conv2_4 f', layers[5].W[4])
	logs.image('conv2_5 f', layers[5].W[5])
	logs.image('conv2_6 f', layers[5].W[6])
	logs.image('conv2_7 f', layers[5].W[7])
	logs.image('conv2_8 f', layers[5].W[8])
	logs.image('conv2_9 f', layers[5].W[9])

	logs.image('conv1', layers[3].get_Output())
	logs.image('conv2', layers[7].get_Output())
	logs.image('softmax', layers[10].get_Output())

	if epoch % 10 == 0 and epoch != 0:
		optimizer.net.save("./models/"+nn.name+"_ep"+str(epoch))
		print("Saved model:", "./models/"+nn.name+"_ep"+str(epoch))

def function_for_each_minibatch(epoch, optimizer, loss_train):
	logs.scalar('lr_value', optimizer.lr)
	logs.scalar('train_loss', loss_train)
	for l in layers:
		if 'conv' in l.name or 'fc' in l.name:
			logs.histogram(l.name+"_W", np.concatenate(l.W, axis=0))
			logs.histogram(l.name+"_b", l.b)
			logs.histogram(l.name+"_W_grad", np.concatenate(l.W_grad,axis=0))
			logs.histogram(l.name+"_b_grad", l.b_grad)

# Fit the net! This might take around 40-60 minutes
sgd.fit((train_inputs, train_outputs), func_ep=function_for_each_epoch, func_mb=function_for_each_minibatch)

# Save the model
nn.save("./models/"+nn.name)
print("Saved last model:", "./models/"+nn.name)
