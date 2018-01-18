# White Box Neural Networks

<img src="https://raw.githubusercontent.com/wiki/develask/White-Box-Neural-Networks/wbnn_logo.png" width="200">

This tookit is the result of two Master thesis, and is still under development. For the moment standard multilayer neural networks have been developed, as well as networks with paralell connections and recurrent neural netoworks, LSTMs included.

See our wiki: [link](https://github.com/develask/White-Box-Neural-Networks/wiki)

TODO list:

• Stage 1:
- [X] Minibatching
- [X] Modify the softmax activation function
- [X] Add covolutional layer
- [ ] Let convolutonal layers accept many inputs with different number of filters
- [X] Change the shape order of inputs: (examples, feats)
- [ ] Change the order of inputs and time
- [ ] Improve speed (do inp adition before multiplication)

• Stage 2:
- [ ] RNN with outputs every time step
- [ ] Multiple output layers
- [ ] Convolution step/stride
- [ ] Parameter sharing
- [ ] Implement other optimization techniques
- [ ] Dropout
- [ ] Activation functions (softmax included) out of the Fully_Connected layer
- [ ] Control initializations in a more apropriate way
- [ ] High Level API/Script to create more complex net (recursive recurrences, seq2seq)
- [ ] Pooling Layers (very low priority)

• Stage 3:
- [ ] Comment the code.
- [ ] Reorganize the code.
- [ ] Change variable, class, function, etc. names to make them more representative.
- [ ] Write an exhaustive wiki, with code and maths explanations.

