import os
import numpy as np
import random as rd
import wbnn
import matplotlib.pyplot as plt
colors = ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#a65628", "#f781bf"]

if not os.path.exists('./plots2'):
    os.makedirs('./plots2')


def funct(x):
	return (np.sin((x+9.19)*6)/(x+9.19)**2/6)*30+0.5


step = 0.05
trunc_num = 15

# Here is the major difference between this script and time_series.py. Here an example not only has multiple
# time step at the beginning but also multiple time steps at the end. Therefore one whole time series will be
# put in a single example. 
def generate_examples(num):
	inps = []
	outs = []
	for _ in range(num):
		init = -5 + 0.5*step#rd.random()*step
		inp = funct(np.arange(init, 8, step)) 
		out = inp[trunc_num:]
		inps.append(inp)
		outs.append(out)


	inps = np.asarray(inps)
	inps = [ [ inps[:,i][:,np.newaxis] ] for i in range(inps.shape[1])]

	outs = np.asarray(outs)
	outs = [ [ outs[:,i][:,np.newaxis] ] for i in range(outs.shape[1])]

	return inps, outs

train = generate_examples(5000)

init = wbnn.layers.initializers.RandomNormal(mean=0, stddev=0.75)


# First NN
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
# The btach-size is lower because in one example we have many more gradients than in the previous case.
# The learning rate is also lower because the batch_size is lower and the gradients are less normalized
# (they are always devided by the batch_size). 
sgd1 = wbnn.optimizers.SGD(nn1, batch_size=8, nb_epochs=30, lr_start=0.05, lr_end=0.05) 
sgd1.lista_preds = []

init = -5 + 0.5*step
xs = np.arange(init,10, step)
lista_test = funct(xs).tolist()

def predict_per_epoch(ep, sgd, loss):
	print("Epoch:", ep, "Loss:", loss)
	lista_preds = lista_test[:trunc_num]
	for _ in range(len(lista_test)-trunc_num):
		input_ = lista_preds
		input_ = [ [ np.asarray([[el]]) ] for el in input_]
		lista_preds.append( sgd.net.prop(input_)[-1][0][0,0] )

	sgd.lista_preds.append(lista_preds)

sgd1.fit(train, predict_per_epoch)


# Second NN
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
sgd2 = wbnn.optimizers.SGD(nn2, batch_size=8, nb_epochs=30, lr_start=0.01, lr_end=0.005, clipping=(-2, 2)) # long time gradients: high risk of gradient exploting in LSTM
sgd2.lista_preds = []

sgd2.fit(train, predict_per_epoch)

# print results...
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
	fig.savefig('plots2/predictions_'+str(i)+'.png')   # save the figure to file
	plt.close(fig)




