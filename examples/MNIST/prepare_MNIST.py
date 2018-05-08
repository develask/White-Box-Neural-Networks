'''
Prepare the data directory and the input/output of the neural networks.
'''

import wget, os, struct, gzip
import numpy as np

url = "http://yann.lecun.com/exdb/mnist/"
files = ["train-labels-idx1-ubyte.gz", "train-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz"]

if not os.path.exists('./data'):
    os.makedirs('./data')

for f in files:
	if not os.path.isfile('./data'+f): 
		print("Downloading", f, "...")
		wget.download(url+f, out='./data/'+f)
		print()


def read_mnist_images(filename, dtype=None):
	with gzip.open('./data/'+filename, 'rb') as f:
		magic, number, rows, cols = struct.unpack('>iiii', f.read(16))
		array = np.frombuffer(f.read(), dtype='uint8')
		array = array.reshape((number, rows * cols))
	return array/255


def read_mnist_labels(filename):
	with gzip.open('./data/'+filename, 'rb') as f:
		magic, _ = struct.unpack('>ii', f.read(8))
		array = np.frombuffer(f.read(), dtype='uint8')
	array = array.reshape(array.size, 1)
	return array

for f in files:
	name = "_".join(f.split("-")[:2])
	if 'images' in f:
		img = read_mnist_images(f)
		np.save('./data/'+name, img)
	else:
		lab = read_mnist_labels(f)
		one_hot = np.zeros((lab.shape[0], 10))
		i=0
		for t in lab:
			one_hot[i,t] = 1
			i+=1
		np.save('./data/'+name, one_hot)