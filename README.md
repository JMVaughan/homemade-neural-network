# Python Neural Network API

This project contains my personal neural network implementation.
Coded in Python, **this project only uses Numpy** for numerical computation (**No Tensorflow, Keras, sklearn, etc.**). All elements of the neural network, such as the backward propogation algorithm, have been implemented from scratch.

Currently, this API only provides the utility of fully connected layers, but work towards introducing convolutional layers is in progress. 

## Features

The API includes the following features: 
- The ability to add as many layers as desired
- A variety of parameter intialisations (Small random and Xavier)
- A variety of optimization procedures (Gradient Descent, Momentum, RMSProp, Adam)
- Dropout Regularisation
- Batch Normalisation
- Stochastic Gradient Descent, Batch Gradient Descent and Mini-Batch Gradient Descent
- Various activation functions (Tanh, Relu, Softmax, Sigmoid)
- The ability to save/ load trained weights

## Example Usage
```python
# Build model
model = Network()
model.add_layer(FullyConnected(256, 'relu', dropout_rate=0.5, batch_norm=True))
model.add_layer(FullyConnected(256, 'relu', dropout_rate=0.5, batch_norm=True))
model.add_layer(FullyConnected(10, 'softmax'))

model.load_parameters(r'tmp\400.npy') # Load previously trained parameters if required
# Train model
model.train(X, Y, learning_rate=0.0075, iterations=2000, mini_batch_size=20, optimizer='Adam')
# Save Parameters
model.save_parameters('MNIST_mini_batch.npy')
```

## Performance

To date, this implementation has achieved 97.9% accuracy on the [MNIST Kaggle competition](https://www.kaggle.com/c/digit-recognizer) for recognising hand written digits. 

![alt text](/examples/MNIST/MNIST_predictions.png)
