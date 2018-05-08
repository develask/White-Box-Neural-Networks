import wbnn
import sys
import numpy as np

# Load the NN we want to test. Write its path in the command as the first argument.
nn = wbnn.NN.load(sys.argv[1])

# Load the test
test_inputs = [[np.load("./data/t10k_images.npy")]]
test_outputs = np.load("./data/t10k_labels.npy")


# Print the NN: each of its layers
print("Network layers: ", nn.name)
for layer in nn.prop_order:
	print("\t",layer.name, layer.get_Output_dim(), layer.__save__dict__()[1])
	if isinstance(layer, wbnn.layers.Dropout):
		layer.probability = 0
print()

# run the test inputs acrros the net
predicted = nn.prop(test_inputs)[0][0]

# initialize the confusion matrix
mat = np.zeros((10,10)).astype(int)

# Get the predicted class i.e., the argmax
test_outputs = np.argmax(test_outputs, axis=1)
predicted = np.argmax(predicted, axis=1)

# Fill the confusion matrix
for i in range(test_outputs.shape[0]):
	mat[test_outputs[i], predicted[i]] += 1


# Print all the resutls
print("Confusion Matrix: row(real)/col(predicted)")
print(mat)
print()

print("Some metrics:")
print("\tAccuracy:\t", np.sum(mat.diagonal())/np.sum(mat))







