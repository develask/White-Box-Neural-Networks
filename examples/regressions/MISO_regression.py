'''
Multiple Input Single Output regression. The input, however, will be represented as a single input of size 2.
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import wbnn
import os

plot_dir = "MISO_plots"
if not os.path.exists('./'+plot_dir):
    os.makedirs('./'+plot_dir)

# This is the funciton we will try to model.
def funct(x1, x2):
	y = ((-x2*np.sin(x1)-x1*np.cos(-x2))+10)/20
	return y

# Here we generate training/test examples.
def generate_examples(nb_examples):
	inps1 = np.random.rand( nb_examples, 1 )*10 - 5
	inps2 = np.random.rand( nb_examples, 1 )*10 - 5
	outs = funct(inps1, inps2)
	inps = np.concatenate([inps1, inps2], axis = 1)
	return [[inps]], [[outs]]

# Define the network as usual.
init = wbnn.layers.initializers.RandomNormal(mean=0, stddev=1)

inp = wbnn.layers.Input(2, "input")
h1 = wbnn.layers.Fully_Connected(200, "hidden 1")
h1.setInitializer(init)
ac1 = wbnn.layers.Activation("sigmoid", "ac1")
h2 = wbnn.layers.Fully_Connected(200, "hidden 2")
h2.setInitializer(init)
ac2 = wbnn.layers.Activation("tanh", "ac2")
fc_out = wbnn.layers.Fully_Connected(1, "fc out")
fc_out.setInitializer(init)
ac_out = wbnn.layers.Activation("sigmoid", "out")
loss = wbnn.layers.Loss("mse", "loss")

inp.addNext(h1)
h1.addNext(ac1)
ac1.addNext(h2)
h2.addNext(ac2)
ac2.addNext(fc_out)
fc_out.addNext(ac_out)
ac_out.addNext(loss)

nn = wbnn.NN("2D_regression")
nn.add_inputs(inp)
nn.initialize()

sgd = wbnn.optimizers.SGD(net=nn, batch_size=64, nb_epochs=40, lr_start=0.2, lr_end=0.05)

# Function per epoch. plot both the NN-modeled function at each epoch
colors = ["#377eb8", "#4daf4a", "#e41a1c", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#a65628", "#f781bf"]

plt_nb_points_per_axis = 200
x1 = np.arange(plt_nb_points_per_axis)
x1 = x1 * (10/plt_nb_points_per_axis)
x1 -= 5

x2 = np.arange(plt_nb_points_per_axis)
x2 = x2 * (10/plt_nb_points_per_axis)
x2 -= 5

x1, x2 = np.meshgrid(x1, x2)
x1 = x1.flatten()[:,np.newaxis]
x2 = x2.flatten()[:,np.newaxis]

y_real = funct(x1, x2)
xs = np.concatenate([x1, x2], axis=1)


def func_per_ep(i, sgd, loss):
	print("Epoch:", i+1, "Training Loss:", loss)
	fig = plt.figure()
	y_pred = nn.prop([[xs]])[-1][0][:,0]

	ax = fig.gca(projection='3d')

	surf = ax.plot_surface(x1.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), x2.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), y_pred.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), cmap=cm.coolwarm, vmin=0, vmax=1,
	                       linewidth=0, antialiased=False)

	ax.set_zlim(0, 1)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.title('Epoch: '+str(i+1), loc='left')

	fig.savefig(plot_dir+'/pred_'+str(i+1)+'.png')   # save the figure to file
	plt.close(fig) 

# Additionally, we will also plot separately the real 2D function, with this piece of code.
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x1.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), x2.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), y_real.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), cmap=cm.coolwarm, vmin=0, vmax=1,
                       linewidth=0, antialiased=False)

ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("Real", loc='left')

fig.savefig(plot_dir+'/real.png')   # save the figure to file
plt.close(fig) 

# Run the function before starting to train, to see how the prediction is with random parameters.
func_per_ep(-1, sgd, "No training yet")

# Train
ex = generate_examples(200000)
print("Training...")
sgd.fit(ex, func_per_ep)




