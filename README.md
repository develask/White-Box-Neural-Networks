# White Box Neural Networks

<img src="https://raw.githubusercontent.com/wiki/develask/White-Box-Neural-Networks/wbnn_logo.png" width="200">

This tookit is the result of two Master thesis, and is still under development. For the moment standard multilayer neural networks have been developed, as well as networks with paralell connections and recurrent neural netoworks, LSTMs included.

For more detailed information:
### [WBNN wiki](https://github.com/develask/White-Box-Neural-Networks/wiki)

## Downloading and installing WBNN

To download this tookit first clone the GitHub project. This can be done via the online interface or running the command below in the folder you want to keep the code:

```bash
cd /path/to/folder/of/WBNN
git clone git@github.com:develask/White-Box-Neural-Networks.git
```

Then you might want to export its path, adding this line to the `~/.bashrc` file:

```bash
EXPORT PYTHONPATH=$PYTHONPATH:"/path/to/folder/of/WBNN/wbnn"
```


TODO list:

• Stage 1:
- [X] Minibatching
- [X] Modify the softmax activation function
- [X] Add covolutional layer
- [X] Let convolutonal layers accept many inputs with different number of filters
- [X] Change the shape order of inputs: (examples, feats)

• Stage 2:
- [X] RNN with outputs every time step
- [X] Multiple output layers
- [X] Convolution step/stride
- [X] Parameter sharing
- [X] Implement other optimization techniques
- [X] Dropout
- [X] Activation functions (softmax included) out of the Fully_Connected layer
- [X] Control initializations in a more apropriate way
- [X] Pooling Layers (very low priority)

• Stage 3:
- [ ] Comment the code.
- [ ] Reorganize the code.
- [ ] Change variable, class, function, etc. names to make them more representative.
- [ ] Write an exhaustive wiki, with code and maths explanations.

