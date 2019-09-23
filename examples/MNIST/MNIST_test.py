import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.model import Network

from src.fullyconnected import FullyConnected

# Import data
data = pd.read_csv("MNIST.csv")

# One-hot encode target values
Y_test = pd.get_dummies(data['label']).as_matrix()

X_test = data.drop(['label'], axis=1).as_matrix()

# Use data not trained on
Y_test = Y_test[:10000, :].T
X_test = X_test[:10000, :].T

# Normalize Data
X_test = (X_test - 33.4089111698)
X_test = X_test/78.6777397608


# Number of examples
m = X_test.shape[1]

# Create model as trained
model = Network()
model.add_layer(FullyConnected(256, 'relu'))
model.add_layer(FullyConnected(256, 'relu'))
model.add_layer(FullyConnected(10, 'softmax'))

# Load parameters
model.load_parameters(r'tmp\400.npy')

print("Data Set Accuracy: {}%".format(model.get_accuracy(X_test, Y_test)))

# Test on random examples
while True:

    example1 = X_test[:, np.random.randint(0, m)].reshape((784, 1))
    example2 = X_test[:, np.random.randint(0, m)].reshape((784, 1))
    example3 = X_test[:, np.random.randint(0, m)].reshape((784, 1))
    example4 = X_test[:, np.random.randint(0, m)].reshape((784, 1))
    example5 = X_test[:, np.random.randint(0, m)].reshape((784, 1))
    example6 = X_test[:, np.random.randint(0, m)].reshape((784, 1))
    example7 = X_test[:, np.random.randint(0, m)].reshape((784, 1))
    example8 = X_test[:, np.random.randint(0, m)].reshape((784, 1))

    ax1 = plt.subplot(241)
    ax2 = plt.subplot(242)
    ax3 = plt.subplot(243)
    ax4 = plt.subplot(244)
    ax5 = plt.subplot(245)
    ax6 = plt.subplot(246)
    ax7 = plt.subplot(247)
    ax8 = plt.subplot(248)


    def get_prediction_as_int(arr):

        arr[arr == max(arr)] = 1
        arr[arr != max(arr)] = 0
        return np.where(arr == 1)[0][0]


    prediction1 = get_prediction_as_int(model.predict(example1))
    prediction2 = get_prediction_as_int(model.predict(example2))
    prediction3 = get_prediction_as_int(model.predict(example3))
    prediction4 = get_prediction_as_int(model.predict(example4))
    prediction5 = get_prediction_as_int(model.predict(example5))
    prediction6 = get_prediction_as_int(model.predict(example6))
    prediction7 = get_prediction_as_int(model.predict(example7))
    prediction8 = get_prediction_as_int(model.predict(example8))

    ax1.imshow(example1.reshape((28, 28)), cmap='gray_r')
    ax1.set_title(str("Prediction: {}".format(prediction1)))

    ax2.imshow(example2.reshape((28, 28)), cmap='gray_r')
    ax2.set_title(str("Prediction: {}".format(prediction2)))

    ax3.imshow(example3.reshape((28, 28)), cmap='gray_r')
    ax3.set_title(str("Prediction: {}".format(prediction3)))

    ax4.imshow(example4.reshape((28, 28)), cmap='gray_r')
    ax4.set_title(str("Prediction: {}".format(prediction4)))

    ax5.imshow(example5.reshape((28, 28)), cmap='gray_r')
    ax5.set_title(str("Prediction: {}".format(prediction5)))

    ax6.imshow(example6.reshape((28, 28)), cmap='gray_r')
    ax6.set_title(str("Prediction: {}".format(prediction6)))

    ax7.imshow(example7.reshape((28, 28)), cmap='gray_r')
    ax7.set_title(str("Prediction: {}".format(prediction7)))

    ax8.imshow(example8.reshape((28, 28)), cmap='gray_r')
    ax8.set_title(str("Prediction: {}".format(prediction8)))

    plt.show()
