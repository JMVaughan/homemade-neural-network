import time
import signal
import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
import os


class Network:
    layer_list = []  # Collection of layers
    layer_count = 0  # Layer number
    output_n = None  # Nodes in last layer
    x = None  # Training examples
    y = None  # Training targets
    mini_batch_size = None  # No. of training examples in mini-batch
    x_n = None  # No. of features
    a_l = None  # last layer activation
    example_n = None  # Total no. of training examples
    step_count = 0  # Count of steps

    cost_list = []  # Collect costs for graphing
    epoch_count = 0  # Iteration number

    parameters_loaded = False

    def add_layer(self, layer):
        """ Add layer to neural network """
        # Input into layer = output of previous layer unless it's the first layer
        if len(self.layer_list) > 0:
            layer.input_n = self.layer_list[-1].output_n
        else:
            layer.input_n = None

        layer.id = self.layer_count

        # Add to list of layers
        self.layer_list.append(layer)

        self.layer_count += 1
        self.output_n = layer.output_n

    def train(self, x, y, learning_rate, iterations, mini_batch_size=None, optimizer='adam'):
        """ Start training. Note: X, Y = (example_dims, # of examples) """
        self.allow_save_with_keyboard_interrupt()
        # Start logging
        logging.basicConfig(filename='training.log', level=logging.INFO, format='%(message)s')

        self.x = x
        self.y = y

        self.x_n = x.shape[0]
        self.example_n = x.shape[1]

        assert x.shape != (0, 0), 'x is empty'
        assert y.shape[0] == self.output_n, 'The number of output nodes does not match first dimension of Y'
        assert y.shape[1] == self.example_n, 'The number of examples in Y and X do not match'

        # Initialize the layers
        self.initialize_layers(self.x_n, optimizer)
        print('Starting training...')
        # Start log
        self.log_training_start(mini_batch_size, learning_rate)
        # Start training
        self.run_training(x, y, learning_rate, iterations, mini_batch_size)
        # Calculate accuracy at the end of training
        accuracy = self.get_accuracy(self.x, self.y)
        print('Training Set Accuracy: {}%'.format(accuracy))
        self.log_training_finish(accuracy)

    def initialize_layers(self, x_n, optimizer):
        """ Initialize parameters """
        # Update first layer input based on dimensions of X
        self.layer_list[0].input_n = x_n

        for l in self.layer_list:
            l.initialize_layer(optimizer)

        self.layer_list[-1].last = True

    def run_training(self, x, y, learning_rate, iterations, mini_batch_size):
        """ Run training with mini-batches """
        self.mini_batch_size = mini_batch_size
        # Run training
        for i in range(1, iterations + 1):
            self.shuffle_data(x, y)
            running_cost = 0
            count = 0
            # Get mini-batch
            for x_batch, y_batch in self.batch_generator(x, y, mini_batch_size):
                # Perform training step
                running_cost += self.step(x_batch, y_batch, learning_rate)
                count += 1

            # Average the cost over every batch in epoch
            average_cost = running_cost/count
            # Gather costs
            self.cost_list.append(average_cost)
            self.epoch_count += 1

            # Print cost and accuracy
            self.print_cost(average_cost, i, iterations, calculate_accuracy=False)

            # Save weights every 200 iterations
            if i % 200 == 0:
                if not os.path.isdir('tmp'):
                    os.mkdir('tmp/')
                self.save_parameters(r'tmp/{}.npy'.format(i))

        # Plot costs
        self.plot_cost()

    def shuffle_data(self, x, y):
        """ Shuffle data inplace """
        # Shuffle data
        shuffled_idx = self.shuffle_indices(self.example_n)
        for j in np.arange(x.shape[0]):
            x[j, :] = x[j, shuffled_idx]

        for j in np.arange(y.shape[0]):
            y[j, :] = y[j, shuffled_idx]

    @staticmethod
    def shuffle_indices(m):
        """ Get a shuffled list of indices """
        shuffled_idx = np.arange(m)
        np.random.shuffle(shuffled_idx)
        return shuffled_idx

    @staticmethod
    def batch_generator(x, y, mini_batch_size):
        """ Generate mini-batches """
        example_n = x.shape[1]
        for i in range(0, example_n, mini_batch_size):
            # Handle case when mini-batch is smaller than requested size
            if example_n - i < mini_batch_size:
                random_idx = np.random.randint(0, i, i + mini_batch_size - example_n)
                yield np.concatenate((x[:, i:i + mini_batch_size], x[:, random_idx]), axis=1), \
                      np.concatenate((y[:, i:i + mini_batch_size], y[:, random_idx]), axis=1)
            else:
                yield x[:, i:i + mini_batch_size], y[:, i:i + mini_batch_size]

    def step(self, x, y, learning_rate):
        """ Perform a single step """
        self.step_count += 1
        # 1. Forward propagation
        a_l = self.propagate_forwards(x)
        # 2. Calculate cost
        cost = self.compute_cost(a_l, y)
        # 3. Backward propagation
        self.propagate_backwards(a_l, x, y)
        # 4. Gradient descent (Update parameters)
        self.update_parameters(learning_rate)

        # Store cost for plotting
        return cost

    def propagate_forwards(self, x, train=True):
        """ Forward propagate through all layers """
        #  Forward Propagation
        a_previous = x
        for i in range(self.layer_count):
            # Get layer
            current_layer = self.layer_list[i]
            # Pass previous activation through current layer
            a_current = current_layer.linear_activation_forward(a_previous, train)
            # Copy current activation for next layer
            a_previous = np.array(a_current, copy=True)

        return a_current

    def propagate_backwards(self, a_l, x, y):
        """ Perform backwards propagation for categorical cross-entropy"""
        # Back-prop through last layer
        next_layer_da = self.layer_list[-1].back_prop(mini_batch_size=self.mini_batch_size, a_l=a_l, y=y)
        # Iterate backwards through all layers except last
        for i in reversed(range(self.layer_count - 1)):
            layer = self.layer_list[i]
            next_layer_da = layer.back_prop(da_output=next_layer_da, mini_batch_size=self.mini_batch_size)

            assert layer.dz.shape == layer.z.shape
            assert layer.dw.shape == layer.w.shape
            assert layer.db.shape == layer.b.shape

    def update_parameters(self, learning_rate):
        """ Update parameters in each of the layers"""
        for l in self.layer_list:
            l.update_parameters(learning_rate)

    def predict(self, x):
        """ Make prediction """
        return self.propagate_forwards(x, False)

    def get_accuracy(self, x, y):
        """ Calculate accuracy on passed-in data set """
        accuracy = np.sum(np.where(np.count_nonzero(np.round(
            self.predict(x)) - y, axis=0) == 0, 1, 0)) * 100 / x.shape[1]
        return accuracy

    def compute_cost(self, a_l, y):
        """ Use cost function as defined by last layer """
        return self.layer_list[-1].activation_function.cost_function(self.mini_batch_size, a_l, y)

    def print_cost(self, cost, i, iterations, calculate_accuracy=True):
        """ Print current cost, iterations, and percentage complete """
        if calculate_accuracy and i % 100 == 0:
            print('Training Set Accuracy: {}%'.format(self.get_accuracy(self.x, self.y)))
        if i % 10 == 0 and i > 0:
            print("{}: {:.1f}%: {}".format(i, (100*i/iterations), cost))

    def plot_cost(self):
        """ Plot cost against iterations """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(self.epoch_count), self.cost_list)
        ax.set_title("Cost over training period")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")
        plt.show()

    def save_parameters(self, file="temp.npy"):
        """ Save parameters to file """
        data = {}
        for i, l in enumerate(self.layer_list):
            key_w = "{}_W".format(str(i))
            key_b = "{}_B".format(str(i))
            data[key_w] = l.w
            data[key_b] = l.b

            if l.batch_norm:
                key_gamma = "{}_GAMMA".format(str(i))
                key_beta = "{}_BETA".format(str(i))
                data[key_gamma] = l.gamma
                data[key_beta] = l.beta

                key_mu = "{}_MU".format(str(i))
                key_var = "{}_VAR".format(str(i))
                data[key_mu] = l.weighted_avg_dict['mu']
                data[key_var] = l.weighted_avg_dict['var']

        np.save(file, data)

    def load_parameters(self, file):
        """ Load previously saved parameters from file """
        parameters = np.load(file)
        parameter_dict = parameters[()]
        for key, array in parameter_dict.items():
            layer_number, param_type = key.split("_")  # param_type = W or B
            layer = self.layer_list[int(layer_number)]
            layer.set_parameter(array, param_type)
            layer.parameters_loaded = True

    def signal_handler(self, signal, frame):
        """ Intercept ctrl+c command to allow weight save """
        accuracy = self.get_accuracy(self.x, self.y)
        self.log_training_finish(accuracy)
        print('Training Set Accuracy: {}%'.format(accuracy))
        file = input("Please enter a <your_file.npy> to save parameters, or press q to exit: ")
        if file != "q":
            self.save_parameters(file)
        sys.exit(0)

    def allow_save_with_keyboard_interrupt(self):
        """ Catch CTRL+C """
        signal.signal(signal.SIGINT, self.signal_handler)

    def check_grads(self, x, y):
        """ This function can be called to ensure that implementation of back propagation is correct"""
        epsilon = 10 ** (-7)
        x_n = x.shape[0]
        self.mini_batch_size = x.shape[1]
        self.initialize_layers(x_n, optimizer='GradientDescent')

        # STEP 1: Calculate numerical_grads
        print('Calculating numerical gradients...')
        numerical_grads = self.get_numerical_grads(epsilon, x, y)

        # STEP 2: Calculate back_prop_grads and gather into vector
        print('Calculating back prop back_prop_grads...')
        self.propagate_backwards(self.propagate_forwards(x), x, y)
        back_prop_grads = self.get_back_prop_grads()

        # STEP 3: Calculate difference (using norms)
        print('Calculating difference...')
        numerator = np.linalg.norm(numerical_grads - back_prop_grads)
        denominator = np.linalg.norm(numerical_grads) + np.linalg.norm(back_prop_grads)
        difference = numerator/denominator
        print("Difference: {}\nEpsilon: {}".format(difference, epsilon))

    def get_numerical_grads(self, epsilon, x, y):
        parameter_total = self.get_parameter_total()

        # Create storage for numerical_grads
        numerical_grads = np.zeros(shape=(parameter_total, 1))
        # Update numerical_grads
        count = 0

        def calc_numerical_grad(parameter, i, j):
            parameter[i, j] += epsilon
            j_plus = self.compute_cost(self.propagate_forwards(x), y)
            parameter[i, j] -= 2 * epsilon
            j_minus = self.compute_cost(self.propagate_forwards(x), y)
            # Put un-perturb weights
            parameter[i, j] += epsilon
            return (j_plus - j_minus) / (2 * epsilon)

        for l, layer in enumerate(self.layer_list):
            # W
            for i in np.arange(layer.w.shape[0]):
                for j in np.arange(layer.w.shape[1]):
                    numerical_grads[count] = calc_numerical_grad(layer.w, i, j)
                    count += 1
            # B
            for i in np.arange(layer.b.shape[0]):
                numerical_grads[count] = calc_numerical_grad(layer.b, i, 0)
                count += 1

            if layer.batch_norm:
                # Gamma
                for i in np.arange(layer.gamma.shape[0]):
                    numerical_grads[count] = calc_numerical_grad(layer.gamma, i, 0)
                    count += 1

                # Beta
                for i in np.arange(layer.beta.shape[0]):
                    numerical_grads[count] = calc_numerical_grad(layer.beta, i, 0)
                    count += 1

        return numerical_grads

    def get_parameter_total(self):
        parameter_total = 0
        for layer in self.layer_list:
            # W and B
            parameter_total += layer.input_n*layer.output_n + layer.output_n
            if layer.batch_norm:
                # Gamma and Beta
                parameter_total += 2*layer.output_n
        return parameter_total

    def get_back_prop_grads(self):
        # Gather all back_prop_grads
        back_prop_grads = np.array([])
        for layer in self.layer_list:
            back_prop_grads = np.concatenate((back_prop_grads, layer.dw.flatten()), axis=0)
            back_prop_grads = np.concatenate((back_prop_grads, layer.db.flatten()), axis=0)
            if layer.batch_norm:
                back_prop_grads = np.concatenate((back_prop_grads, layer.dgamma.flatten()), axis=0)
                back_prop_grads = np.concatenate((back_prop_grads, layer.dbeta.flatten()), axis=0)
        back_prop_grads = np.reshape(back_prop_grads, newshape=(back_prop_grads.shape[0], 1))

        return back_prop_grads

    def log_training_start(self, mini_batch_size, learning_rate):
        # Log training start
        logging.info('Start: {}\n'.format(time.strftime('%d %b %y %H:%M:%S')))
        template = '\t{0:9}|{1:20}|{2:13}|{3:14}|{4:9}|{5:25}'.rjust(6)
        logging.info(template.format('Layer no.', 'Activation Function', 'Hidden Units', 'Dropout Value', 'Optimizer', 'Parameter Initialization'))
        for i, layer in enumerate(self.layer_list):
            if self.parameters_loaded:
                param_initialization = 'Loaded'
            else:
                param_initialization = layer.parameter_initialization
            logging.info(template.format(i, type(layer).__name__, layer.output_n, layer.dropout_rate, layer.optimizer_string, param_initialization))
        logging.info('\tLearning Rate: {}'.format(learning_rate).rjust(6))
        logging.info('\tMini-Batch Size: {}'.format(mini_batch_size).rjust(6))
        logging.info('\tNo. of Examples: {}'.format(self.example_n).rjust(6))
        logging.info('\tNo. of Features: {}'.format(self.x_n).rjust(6))
        logging.info('\n')

    def log_training_finish(self, accuracy):
        # Log training start
        logging.info('Finish: {}'.format(time.strftime('%d %b %y %H:%M:%S')))
        logging.info('\tEpochs: {}'.rjust(6).format(self.epoch_count))
        logging.info('\tTraining Set Accuracy: {}%'.rjust(6).format(accuracy))
        logging.info('\n')
