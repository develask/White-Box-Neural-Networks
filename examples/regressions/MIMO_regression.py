'''
This is the last regression example. Now we have two separated float inputs and two different output
layers. 
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import wbnn
import os

plot_dir = "MIMO_plots"
if not os.path.exists('./'+plot_dir):
    os.makedirs('./'+plot_dir)

# The first function we want to model.
def f1(x1, x2):
	y = ((-x2*np.sin(x1)-x1*np.cos(-x2))+10)/20
	return y

# The second one.
def f2(x1, x2):
	y = ((x1/3)**2+(x2/3)**2-(np.cos(3*x1)+np.cos(2*x2)))/10 +0.25
	y = 1-y
	return y

def generate_examples(nb_examples):
	inps1 = np.random.rand( nb_examples, 1 )*10 - 5
	inps2 = 1-(np.random.rand( nb_examples, 1 )*10 - 5)
	outs1 = f1(inps1, inps2)
	outs2 = f2(inps1, inps2)
	return [[inps1, inps2]], [[outs1, outs2]]


# Define the layer in the net.
init = wbnn.layers.initializers.RandomNormal(mean=0, stddev=0.5)

inp1 = wbnn.layers.Input(1, "input1")
inp2 = wbnn.layers.Input(1, "input2")

h1 = wbnn.layers.Fully_Connected(30, "hidden 1")
h1.setInitializer(init)
ac1 = wbnn.layers.Activation("tanh", "ac1")

h2 = wbnn.layers.Fully_Connected(30, "hidden 2")
h2.setInitializer(init)
ac2 = wbnn.layers.Activation("tanh", "ac2")

mix_fc = wbnn.layers.Fully_Connected(30, "mix")
mix_fc.setInitializer(init)
mix_ac = wbnn.layers.Activation("tanh", "ac_mix")

mix_fc2 = wbnn.layers.Fully_Connected(30, "mix")
mix_fc2.setInitializer(init)
mix_ac2 = wbnn.layers.Activation("relu", "ac_mix")

fc_out1 = wbnn.layers.Fully_Connected(1, "fc out1")
fc_out1.setInitializer(init)
ac_out1 = wbnn.layers.Activation("sigmoid", "out1")
loss1 = wbnn.layers.Loss("mse", "loss1")

fc_out2 = wbnn.layers.Fully_Connected(1, "fc out2")
fc_out2.setInitializer(init)
ac_out2 = wbnn.layers.Activation("sigmoid", "out2")
loss2 = wbnn.layers.Loss("mse", "loss2")


# Define the the connections.
inp1.addNext(h1)
h1.addNext(ac1)

inp2.addNext(h2)
h2.addNext(ac2)

ac1.addNext(mix_fc)
ac2.addNext(mix_fc)
mix_fc.addNext(mix_ac)

mix_ac.addNext(mix_fc2)
mix_fc2.addNext(mix_ac2)

mix_ac2.addNext(fc_out1)
fc_out1.addNext(ac_out1)
ac_out1.addNext(loss1)

mix_ac.addNext(fc_out2)
fc_out2.addNext(ac_out2)
ac_out2.addNext(loss2)


# Initialize the net.
nn = wbnn.NN("2D_regression")
nn.add_inputs(inp1, inp2)
nn.initialize()

# Define the optimizer.
sgd = wbnn.optimizers.SGD(nn, batch_size=128, nb_epochs=40, lr_start=0.2, lr_end=0.2)


# function per epoch. Save the two functions modeled by the net in each epoch. 
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

y_real1 = f1(x1, x2)
y_real2 = f2(x1, x2)

def func_per_ep(i, sgd, loss):
	print("Epoch:", i+1, "Training Loss:", loss)
	
	y_pred1 = nn.prop([[x1, x2]])[-1][0][:,0]
	y_pred2 = nn.prop([[x1, x2]])[-1][1][:,0]


	# First Function
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	surf = ax.plot_surface(x1.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), x2.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), y_pred1.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), cmap=cm.coolwarm, vmin=0, vmax=1,
	                       linewidth=0, antialiased=False)

	ax.set_zlim(0, 1)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.title('Epoch: '+str(i+1), loc='left')

	fig.savefig(plot_dir+'/pred1_'+str(i+1)+'.png')   # save the figure to file
	plt.close(fig) 


	# Second Function
	fig = plt.figure()
	ax = fig.gca(projection='3d')

	surf = ax.plot_surface(x1.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), x2.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), y_pred2.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), cmap=cm.coolwarm, vmin=0, vmax=1,
	                       linewidth=0, antialiased=False)

	ax.set_zlim(0, 1)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('z')

	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.title('Epoch: '+str(i+1), loc='left')

	fig.savefig(plot_dir+'/pred2_'+str(i+1)+'.png')   # save the figure to file
	plt.close(fig) 


# Now run the same plots but with the real functions, to see the difference.
# First real function
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x1.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), x2.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), y_real1.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), cmap=cm.coolwarm, vmin=0, vmax=1,
                       linewidth=0, antialiased=False)

ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("Real", loc='left')

fig.savefig(plot_dir+'/real1.png')   # save the figure to file
plt.close(fig) 


# Second real function
fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x1.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), x2.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), y_real2.reshape(plt_nb_points_per_axis, plt_nb_points_per_axis), cmap=cm.coolwarm, vmin=0, vmax=1,
                       linewidth=0, antialiased=False)

ax.set_zlim(0, 1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title("Real", loc='left')

fig.savefig(plot_dir+'/real2.png')   # save the figure to file
plt.close(fig) 

# Run the function before starting to train, to see how the prediction is with random parameters.
func_per_ep(-1, sgd, "No training yet")

# Train
ex = generate_examples(350000)
print("Training...")
sgd.fit(ex, func_per_ep)




