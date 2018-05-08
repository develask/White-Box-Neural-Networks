'''
This script downloads the ag_news data and glove word vectors to train a net for text classification.
Here the data is also processed to be formated as needed by the NN.
'''

import numpy as np
import wget
import os
import zipfile
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

if not os.path.exists('./data'):
    os.makedirs('./data')

url = "http://nlp.stanford.edu/data/"
file = "glove.6B.zip"

if not os.path.isfile('./data/'+file):
	print("Downloading:", file)
	wget.download(url+file, out='./data/'+file)
	print()
	print("Extracting:", file)
	zip_ref = zipfile.ZipFile("data/"+file, 'r')
	zip_ref.extractall("data/")
	zip_ref.close()

url = "https://github.com/mhjabreel/CharCNN/raw/master/data/ag_news_csv/"
files = ["test.csv", "train.csv"]

for file in files:
	if not os.path.isfile('./data/'+file):
		print("Downloading:", file)
		wget.download(url+file, out='./data/'+file)
		print()

print("Reading train and test databases...")

def valid_token(tok):
	return tok not in "\"\"''()[]{}<>$%&#-_/"

def tokenize(sentence):
	#return [list(filter(valid_token,word_tokenize(t))) for t in sent_tokenize(sentence)]
	return list(filter(valid_token,word_tokenize(sentence)))

with open('data/train.csv') as f:
	train_csv = [(int(y[0]), tokenize(y[1])) for y in [ x[1:].split('","')[:2] for x in f.read().lower().splitlines()]]

with open('data/test.csv') as f:
	test_csv = [(int(y[0]), tokenize(y[1])) for y in [ x[1:].split('","')[:2] for x in f.read().lower().splitlines()]]


# This parameter sets the dimension of the word-vectors. See data/ folder for posibilities
glove_dim = 50
print("Generating problem specific word-vector matrix...")
print("Reading glove word-vectors: (Dimension: %d)" % glove_dim)
with open('data/glove.6B.%dd.txt'%glove_dim) as f:
	lines = [x.split() for x in f.read().splitlines()]
	words = [x[0] for x in lines]
	vectors = np.asarray([[float(y) for y in x[1:]] for x in lines])

print("Creating a new word-vector matrix for ag_news database. This usually takes several minutes...")
unique_words = ['<PAD>', '<UNK>']
for label, sentence in train_csv:
	for word in sentence:
		if word not in unique_words and word in words:
			unique_words.append(word)
unique_words_matrix = np.zeros((len(unique_words), glove_dim))
unknown_words = []
for i, word in enumerate(unique_words):
	try:
		j = words.index(word)
		unique_words_matrix[i,:] = vectors[j,:]
	except Exception as e:
		unknown_words.append(word)
print("Known words: (%d)"%(len(unique_words)-len(unknown_words)))

with open('data/train_words.txt', 'w') as f:
	f.write('\n'.join(unique_words))
np.save('data/train_embeddings_%d'%glove_dim, unique_words_matrix)

words = ['<PAD>', '<UNK>']+words
with open('data/test_words.txt', 'w') as f:
	f.write('\n'.join(words))
test_vectors = np.zeros((len(words), glove_dim))
test_vectors[2:,:] = vectors
np.save('data/test_embeddings_%d'%glove_dim, test_vectors)

def my_index_of(list_, element):
	try:
		return list_.index(element)
	except Exception as e:
		return 1

train_max_t = max([len(x[1]) for x in train_csv])
inputs = np.zeros((len(train_csv), train_max_t))
labels = np.zeros((len(train_csv), 4))
for i, ex in enumerate(train_csv):
	inputs[i,-len(ex[1]):] = np.asarray([my_index_of(unique_words, x) for x in ex[1]])
	labels[i, ex[0]-1] = 1
np.save('data/train_inputs', inputs)
np.save('data/train_labels', labels)

test_max_t = max([len(x[1]) for x in test_csv])
inputs = np.zeros((len(test_csv), test_max_t))
labels = np.zeros((len(test_csv), 4))
for i, ex in enumerate(test_csv):
	inputs[i,-len(ex[1]):] = np.asarray([my_index_of(words, x) for x in ex[1]])
	labels[i, ex[0]-1] = 1
np.save('data/test_inputs', inputs)
np.save('data/test_labels', labels)


