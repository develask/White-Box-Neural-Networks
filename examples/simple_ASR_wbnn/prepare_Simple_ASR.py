'''
Download the data, and compute the spectograms of the audios to feed them to the NNs.
'''



import wget, os
import librosa
from scipy.io import wavfile
import numpy as np
import pickle
import tarfile

if not os.path.exists('./data'):
	os.makedirs('./data')



url = "http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz"
f = "speech_commands_v0.01.tar.gz"

if not os.path.isfile('./data/'+f): 
	print("Downloading:", f, "...")
	wget.download(url, out='./data/'+f)
	print("\nDownloaded")

	print("Uncompressing...")
	tar = tarfile.open("data/"+f, "r:gz")
	tar.extractall("data/")
	tar.close()
	print("Uncompressed")



def spectogram(audio_file_name):
	sample_rate, samples = wavfile.read(audio_file_name) # load the data
	samples = (samples+65536)/2/65536 - 0.5
	spectogram = librosa.feature.melspectrogram(y=samples, sr=int(sample_rate))
	return np.log(spectogram+0.00001)


train_spectograms = []
train_labels = [[[]]]
test_spectograms = []
test_labels = [[[]]]

for _ in range(32):   # 32 time-steps = 1second of audio
	train_spectograms.append([])
	test_spectograms.append([])


dirs = ["data/up/", "data/down/", "data/right/", "data/left/"] # The classes we want to predict

def id_to_onehot(id_, len_):
	onehot = np.zeros(len(dirs))
	onehot[id_] = 1
	return onehot 

for id_, dir_ in enumerate(dirs):
	kont = 0 # to keep 300 in test
	print("Preparing:", dir_)
	for f in os.listdir(dir_):
		s = spectogram(dir_+f)
		if s.shape[1]==32:
			if kont<300: # to test partition
				for t in range(32):
					test_spectograms[t].append(s[:,t].tolist())
				test_labels[0][0].append(id_to_onehot(id_, len(dirs)))
			else:
				for t in range(32):
					train_spectograms[t].append(s[:,t].tolist())
				train_labels[0][0].append(id_to_onehot(id_, len(dirs)))
			kont += 1


for t in range(32):
	train_spectograms[t] = [ np.asarray(train_spectograms[t]) ]
	test_spectograms[t] = [ np.asarray(test_spectograms[t]) ]

train_labels[0][0] = np.asarray(train_labels[0][0])
test_labels[0][0] = np.asarray(test_labels[0][0])


with open('data/train_spectograms.pickle', 'wb') as f:
	pickle.dump(train_spectograms, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/test_spectograms.pickle', 'wb') as f:
	pickle.dump(test_spectograms, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/train_labels.pickle', 'wb') as f:
	pickle.dump(train_labels, f, protocol=pickle.HIGHEST_PROTOCOL)
with open('data/test_labels.pickle', 'wb') as f:
	pickle.dump(test_labels, f, protocol=pickle.HIGHEST_PROTOCOL)


