# White Box Neural Networks

![wbnn_logo](https://raw.githubusercontent.com/wiki/develask/White-Box-Neural-Networks/wbnn_logo.png | width=100)

This tookit is the result of two Master thesis, and is still under development. For the moment standard multilayer neural networks have been developed, as well as networks with paralell connections and recurrent neural netoworks, LSTMs included.

See our wiki: [link](https://github.com/develask/White-Box-Neural-Networks/wiki)

Next steps:

• Stage 1:
- [X] Minibatching
- [X] Modify the softmax activation function
- [X] Add covolution layer
- [X] Change the shape order of inputs: (examples, feats)
- [ ] Change the order of inputs and time
- [ ] Improve speed (do inp adition before multiplication)

• Stage 2:
- [ ] Pooling Layers
- [ ] Dropout
- [ ] Convolution step
- [ ] RNN with outputs every time step
- [ ] Implement other optimization techniques
- [ ] Activation functions (softmax included) out of the Fully_Connected layer
- [ ] Control initializations in a more apropriate way
- [ ] High Level API/Script to create more complex net (recursive recurrences, seq2seq)

• Stage 3:
- [ ] Comment the code.
- [ ] Reorganize the code.
- [ ] Change variable, class, etc. names to make it more clear.
- [ ] Write an exhaustive wiki, with code and maths explanations.

