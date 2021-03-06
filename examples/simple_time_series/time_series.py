'''
Here we will work with time series. Given a series of consecutive points of a function (funct(x)), we will try to predict
the next point. There are two scripts. In this one we will only take into account the trunc_num latest points to do the
prediction. In the time_series_2.py script the input will be all the series since the begining.
'''

import os
import numpy as np
import random as rd
import wbnn
import matplotlib.pyplot as plt
colors = ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#a65628", "#f781bf"]

if not os.path.exists('./plots'):
    os.makedirs('./plots')


# This is the function we will be working with
def funct(x):
	return (np.sin((x+9.19)*6)/(x+9.19)**2/6)*30+0.5


# Step is used to discretize the function.
step = 0.05
trunc_num = 15


# This function generates training/test examples ready to be fed to the net. 
def generate_examples(num):
	inps = []
	outs = []
	for _ in range(num):
		init = -5 + rd.random()*step
		lista = funct(np.arange(init,8, step)) # We train from the x ~5 to x=8 of the series. We will test from
		                                       # x ~5 to x=10
		for i2 in range(lista.shape[0]-trunc_num):
			inps.append( lista[i2:i2+trunc_num] )
			outs.append( lista[i2+trunc_num] )
	
	inps = np.asarray(inps)
	inps = [ [ inps[:,i][:,np.newaxis] ] for i in range(inps.shape[1])]
	outs = [[np.asarray(outs)[:, np.newaxis]]]

	return inps, outs

# We will train with 500 rounds of examples.
train = generate_examples(500)

# set the variable initializers of some layers.
init = wbnn.layers.initializers.RandomNormal(mean=0, stddev=1)


# We will try with two RNNs. The first one is based on sigmoidal activation units.
inp = wbnn.layers.Input(1, "input")
h1 = wbnn.layers.Fully_Connected(20, "h1")
h1.setInitializer(init)
ac1 = wbnn.layers.Activation("sigmoid", "ac1")
out = wbnn.layers.Fully_Connected(1, "out")
out.setInitializer(init)
ac_out = wbnn.layers.Activation("sigmoid", "ac3")
loss = wbnn.layers.Loss("mse", "loss")


inp.addNext(h1)
h1.addNext(ac1)
ac1.addNext(out)
out.addNext(ac_out)
ac_out.addNext(loss)

ac1.addNextRecurrent(h1)

nn1 = wbnn.NN("sigmoid_recurrent")
nn1.add_inputs(inp)
nn1.initialize()

print("Training sigmoid-RNN...")
sgd1 = wbnn.optimizers.SGD(net=nn1, batch_size=256, nb_epochs=15, lr_start=1, lr_end=1)
sgd1.lista_preds = [] # We initialize this list to save the predictions of the series in each epoch

# We save a prediction of the series in each epoch.
init = -5 + 0.5*step
xs = np.arange(init,10, step)
lista_test = funct(xs).tolist()

# We print the loss and also predict a whole series. For that we give the NN the first
# 15 points of the real series and then feedback the output of the net to the input.
def predict_per_epoch(ep, sgd, loss):
	print("Epoch:", ep, "Loss:", loss)
	lista_preds = lista_test[:trunc_num]
	for _ in range(len(lista_test)-trunc_num):
		input_ = lista_preds[-15:]
		input_ = [ [ np.asarray([[el]]) ] for el in input_]
		lista_preds.append( sgd.net.prop(input_)[-1][0][0,0] )
	sgd.lista_preds.append(lista_preds)

# Train
sgd1.fit(train, predict_per_epoch)


# Second NN. LSTM based
init = wbnn.layers.initializers.RandomNormal(mean=0, stddev=0.5)
inp = wbnn.layers.Input(1, "input")
lstm1 = wbnn.layers.LSTM(20, "lstm1")
lstm1.setInitializer(init)
out = wbnn.layers.Fully_Connected(1, "out")
out.setInitializer(init)
ac_out = wbnn.layers.Activation("sigmoid", "ac_out")
loss = wbnn.layers.Loss("mse", "loss")


inp.addNext(lstm1)
lstm1.addNext(out)
out.addNext(ac_out)
ac_out.addNext(loss)

nn2 = wbnn.NN("LSTM_RNN")
nn2.add_inputs(inp)
nn2.initialize()

print()
print("Training LSTM-RNN...")
sgd2 = wbnn.optimizers.SGD(net=nn2, batch_size=256, nb_epochs=15, lr_start=1, lr_end=1)
sgd2.lista_preds = []

sgd2.fit(train, predict_per_epoch)

# Once we have saved the  prediction of the time series in each epcoh, we can plot them:
xs = xs.tolist()

for i in range(len(sgd1.lista_preds)):
	pred1 = sgd1.lista_preds[i]
	pred2 = sgd2.lista_preds[i]

	fig = plt.figure()
	lengends = []
	lengends.append(plt.plot(xs, lista_test, label='Real', marker=None, color=colors[1]))
	lengends.append(plt.plot(xs, pred1, label='Predicción RNN-sigm', marker=None, color=colors[2]))
	lengends.append(plt.plot(xs, pred2, label='Predicción LSTM', marker=None, color=colors[3]))
	lengends = [l[0] for l in lengends]

	plt.legend(handles=lengends,loc=1)
	plt.ylim((0,1))

	plt.ylabel('f(x)')
	plt.xlabel('x')
	plt.title('Epoch: '+str(i), loc='left')
	fig.savefig('plots/predictions_'+str(i)+'.png')   # save the figure to file
	plt.close(fig)




