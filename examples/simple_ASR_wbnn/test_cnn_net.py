import wbnn
import sys
import pickle
import numpy as np

# Load the model we want to test, from the command
nn = wbnn.NN.load(sys.argv[1])

# Open and format the test examples
with open('data/test_spectograms.pickle', 'rb') as handle:
	test_inputs = pickle.load(handle)
with open('data/test_labels.pickle', 'rb') as handle:
	test_outputs = pickle.load(handle)

test_outputs = test_outputs[0][0]

test_inputs = [x for y in test_inputs for x in y]
test_inputs = [[np.concatenate(test_inputs, axis=1)]]


# print the structure fo the NN that is being tested
print("Network layers: ", nn.name)
for layer in nn.prop_order:
	print("\t",layer.name, layer.get_Output_dim(), layer.__save__dict__()[1])
	if isinstance(layer, wbnn.layers.Dropout):
		layer.probability = 0

# Run the test inputs throughout the net
predicted = nn.prop(test_inputs)[0][0]
nb_classes = 4

# Initialize the confusion matrix
mat = np.zeros((nb_classes,nb_classes)).astype(int)

# Get the predicted and real classes
test_outputs = np.argmax(test_outputs, axis=1)
predicted = np.argmax(predicted, axis=1)

# Fill the confusion matrix
for i in range(test_outputs.shape[0]):
	mat[test_outputs[i], predicted[i]] += 1

# Print the results
print("Confusion Matrix: row(real)/col(predicted)")
print(mat)
print()
print("Some metrics:")
print("\tAccuracy:\t", np.sum(mat.diagonal())/np.sum(mat)) # This should be around 90%
