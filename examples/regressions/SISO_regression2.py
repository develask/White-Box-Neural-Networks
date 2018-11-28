'''
In these three scripts, we will be doing regression in WBNN. We will be working with single input/single outputs;
multiple input/single output; multiple input/multiple/output.
'''

import numpy as np
import matplotlib.pyplot as plt
import wbnn
import os

plot_dir = "SISO_plots2"
if not os.path.exists('./'+plot_dir):
    os.makedirs('./'+plot_dir)

# This is the funciton we will try to model.
def funct(x):
	y = np.sin(x)*0.4 + 0.5
	return y

# Here we generate training/test examples.
def generate_examples(nb_examples):
	inps = np.random.rand( nb_examples, 1 )*10 - 5
	inps = np.arange(-5, 5, 10/nb_examples).reshape(nb_examples,1)
	outs = funct(inps) #+ (np.random.rand( nb_examples, 1 ) - 0.5) / 5
	inps = inps + np.random.uniform(0.2, 0.5,  size=( nb_examples, 1 ) ) * ((np.random.randint(2,size=( nb_examples, 1 ) ) -0.5) * 2  )
	return [[inps]], [[outs]]

# Define the network as usual.
init = wbnn.layers.initializers.RandomNormal(mean=0, stddev=1)

inp = wbnn.layers.Input(1, "input")
h1 = wbnn.layers.Fully_Connected(10, "hidden 1")
h1.setInitializer(init)
ac1 = wbnn.layers.Activation("tanh", "ac1")
h2 = wbnn.layers.Fully_Connected(10, "hidden 2")
h2.setInitializer(init)
ac2 = wbnn.layers.Activation("tanh", "ac2")
h3 = wbnn.layers.Fully_Connected(10, "hidden 2")
h3.setInitializer(init)
ac3 = wbnn.layers.Activation("tanh", "ac2")
fc_out = wbnn.layers.Fully_Connected(1, "fc out")
fc_out.setInitializer(init)
ac_out = wbnn.layers.Activation("sigmoid", "out")
loss = wbnn.layers.Loss("mse", "loss") # The loss is now mse.

inp.addNext(h1)
h1.addNext(ac1)
ac1.addNext(h2)
h2.addNext(ac2)
ac2.addNext(h3)
h3.addNext(ac3)
ac3.addNext(fc_out)
fc_out.addNext(ac_out)
ac_out.addNext(loss)

nn = wbnn.NN("1D_regression2")
nn.add_inputs(inp)
nn.initialize()

sgd = wbnn.optimizers.SGD(net=nn, batch_size=1, nb_epochs=1000, lr_start=0.2, lr_end=0.05)

# Function per epoch. plot both the real and the NN-modeled function at each epoch
colors = ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#a65628", "#f781bf"]

xs = np.arange(10000)
xs = xs * 0.001
xs -= 5
y_real = funct(xs)
xs = xs[:, np.newaxis]

results = []


train_dts = generate_examples(20)


berretura = 5
coef = sgd.nb_epochs / (10**berretura)
print_time = [int(coef*x**berretura) for x in range(11)]

def func_per_ep(i, sgd, loss):
	if (i+1) not in print_time:	
		return
	print("Epoch:", i+1, "Training Loss:", loss)
	fig = plt.figure()
	y_pred = nn.prop([[xs]])[-1][0][:,0]
	results.append(y_pred)

	lengends = []
	lengends.append(plt.plot(xs[:,0], y_real, label='Real', marker=None, color=colors[1]))
	lengends.append(plt.plot(xs[:,0], y_pred, label='Prediction', marker=None, color=colors[2]))
	lengends.append(plt.plot(train_dts[0][-1][0][:,0], train_dts[1][-1][0][:,0], label='Samples', marker='x', color=colors[0], linestyle='None'))
	lengends = [l[0] for l in lengends]

	plt.legend(handles=lengends,loc=1)
	plt.ylim((0,1))

	plt.ylabel('f(x)')
	plt.xlabel('x')
	plt.title('Epoch: '+str(i+1), loc='left')
	fig.savefig(plot_dir+'/pred_'+str(i+1)+'.eps')   # save the figure to file
	plt.close(fig) 

func_per_ep(-1, sgd,  "No training yet")
print("Training...")
sgd.fit(train_dts, func_ep=func_per_ep)

