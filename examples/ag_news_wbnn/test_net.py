import wbnn
import sys
import numpy as np
from lookuptable import LookUpTable

# Load the model we want to test
nn = wbnn.NN.load(sys.argv[1])

def convert_inputs(inps):
	return [ [ inps[:,i][:,np.newaxis] ] for i in range(inps.shape[1])]

# Prepare the test examples
test_inputs = np.load("data/test_inputs.npy").astype(np.int32)
test_class = np.load("data/test_labels.npy")

test_inputs = convert_inputs(test_inputs)

# Load the embedding matrix for the text partition
embeddings_test = np.load("data/test_embeddings_50.npy")
# Set the embedding matrix
nn.prop_order[1].lookup = embeddings_test

# Print the NN
print("Network layers: ", nn.name)
for layer in nn.prop_order:
	print("\t",layer.name, layer.get_Output_dim(), layer.__save__dict__()[1])
	if isinstance(layer, wbnn.layers.Dropout):
		layer.probability = 0
print()

# Predict the test exmaples and fill the confusion matrix as usual.
predicted = nn.prop(test_inputs)[-1][0]

mat = np.zeros((4,4)).astype(int)

test_class = np.argmax(test_class, axis=1)
predicted = np.argmax(predicted, axis=1)

for i in range(test_class.shape[0]):
	mat[test_class[i], predicted[i]] += 1


print("Confusion Matrix: row(real)/col(predicted)")
print(mat)
print()

print("Some metrics:")
print("\tAccuracy:\t", np.sum(mat.diagonal())/np.sum(mat))
print()

#######################################################################################
print("###################################################################################")

from nltk.tokenize import word_tokenize
print()
print("Now it's time for you to test the net!")
print("Write the title of the new you want to classify.")
print("The four classes are:")

classes = ["World", "Sports", "Business", "Science & Technology"]
[print("\t",class_) for class_ in classes]
print()

# The next two functions are to separate a sentence into words or tokens.
def valid_token(tok):
	return tok not in "\"\"''()[]{}<>$%&#-_/"

def tokenize(sentence):
	return list(filter(valid_token,word_tokenize(sentence)))

# Now we read all the words "we know".
with open('data/test_words.txt', 'r') as f:
	words = f.read().splitlines()

# For each word we get its index, in the same order that have been used in the embedding.
# If the word is unknown, the index 1 is given: <UNK> token.
def my_index_of(element):
	try:
		return words.index(element)
	except Exception as e:
		return 1

while True:	
	print()
	print("##########################################################################")
	print()
	my_title = input("Write the title (Ctrl + C to quit): \n\t").lower()
	print()

	# Tokenize
	my_title = tokenize(my_title)

	# Make the list of integers (the indexes) and format it according to WBNN.
	my_title = [ [ np.asarray([[my_index_of(w)]]) ] for w in my_title]

	# Get the output of the net
	predicted = nn.prop(my_title)[-1][0]
	predicted_class = np.argmax(predicted)
	
	print("PREDICTED CLASS:", classes[predicted_class])
	print()
	print("Confidence levels in each class:")
	for i in range(len(classes)):
		print("\t%-20s -> %.3f"%(classes[i], predicted[0,i]))


























