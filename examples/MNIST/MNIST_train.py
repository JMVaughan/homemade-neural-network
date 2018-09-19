import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural_network.src.fullyconnected import FullyConnected
from neural_network.src.model import Network

# Get data
data = pd.read_csv("MNIST.csv")
data = data.sample(frac=1)

# One-hot encode target data
Y = pd.get_dummies(data['label']).as_matrix()
X = data.drop(['label'], axis=1).as_matrix()

# Get training set
Y = Y[:2000, :].T
X = X[:2000, :].T

# Normalize
mean = np.mean(X)
std_dev = np.std(X)
X = (X - mean)
X = X/std_dev

# Get no. of features, x_n, and no. of examples, m.
x_n = X.shape[0]
m = X.shape[1]

# Build model
model = Network()
model.add_layer(FullyConnected(256, 'relu', dropout_rate=0.5, batch_norm=True))
model.add_layer(FullyConnected(256, 'relu', dropout_rate=0.5, batch_norm=True))
model.add_layer(FullyConnected(10, 'softmax'))

model.load_parameters(r'tmp\400.npy')
# Train model
model.train(X, Y, learning_rate=0.0075, iterations=2000, mini_batch_size=20, optimizer='Adam')
# Save Parameters
model.save_parameters('MNIST_mini_batch.npy')

# Test on random examples
while True:

    example1 = X[:, np.random.randint(0, m)]
    example2 = X[:, np.random.randint(0, m)]

    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)

    example1 = example1.reshape((784, 1))
    example2 = example2.reshape((784, 1))

    def get_prediction_as_int(arr):
        arr[arr == max(arr)] = 1
        arr[arr != max(arr)] = 0
        return np.where(arr == 1)[0][0]

    prediction1 = get_prediction_as_int(model.predict(example1))
    prediction2 = get_prediction_as_int(model.predict(example2))

    print(prediction1)
    print(prediction2)

    ax1.imshow(example1.reshape((28, 28)), cmap='gray_r')
    ax1.set_title(str("Prediction: {}".format(prediction1)))
    ax2.imshow(example2.reshape((28, 28)), cmap='gray_r')
    ax2.set_title(str("Prediction: {}".format(prediction2)))

    plt.show()