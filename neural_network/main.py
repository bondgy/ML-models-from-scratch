import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Layer:
    """
    Represents an individual layer within the model
    """

    def __init__(self, size, activation=None, inputs=np.array([]), index=None):
        """
        Initializes the individual layer, creating nodes if not supplied
        :param size: the size (amount of nodes) in the layer
        :param activation: the activation function used
        :param inputs: numpy array containing the input nodes
        :param index: the index of the layer in relation to its Layers class
        """
        self.shape = size
        self.activation = activation
        self.index = index
        self.sum = np.array([])
        if inputs.shape[0] > 0:
            self.layer = inputs
            self.layer = np.expand_dims(inputs, axis=1)
        else:
            self.layer = np.random.random_sample((size, 1))

    def set_index(self, index):
        """
        Sets the index. Can also use object.index = value
        :param index: The index value
        :return: None
        """
        self.index = index

    @staticmethod
    def linear_relu(sum_array):
        """
        Performs relu activation
        :param sum_array: array of the sum of linear combinations
        :return: the relu activation output
        """
        if sum_array[0] > 0:
            return sum_array[0]
        else:
            return 0

    @staticmethod
    def e_to_power(sum_array):
        """
        Raises e to the power of the sum of the array
        :param sum_array: array of the sum of the linear combination
        :return: e^sum_array
        """
        return np.power(np.e, sum_array[0])

    @staticmethod
    def linear_softmax(sum_array):
        """
        Performs the softmax activation on the linear combination array output
        :param sum_array: array containing the linear combination output
        :return: The output from the softmax function
        """
        raised_arr = np.power(np.e, sum_array).transpose()
        total_sum = np.sum(raised_arr)
        activations = np.divide(raised_arr, total_sum)
        return activations

    @staticmethod
    def linear_sigmoid(sum_array):
        """
        Performs the sigmoid activation function
        :param sum_array: array containing the linear combination output
        :return: The output from the sigmoid activation
        """
        negative_sum_array = np.multiply(-1, sum_array)
        e_power = np.power(np.e, negative_sum_array)
        denominator = np.add(1, e_power)
        activations = np.divide(1, denominator)
        return activations

    def activate_neurons(self, weights, activation_function):
        """
        Performs the given activation function for the layer
        :param weights: The current weights for the layer
        :param activation_function: The activation function to be used, options being relu, softmax, or sigmoid
        :return: Array containing the output from the linear combination and activation output
        """
        sum_array = np.matmul(self.layer.transpose(), weights)
        sum_array_copy = sum_array.copy()
        if activation_function == 'relu':
            return sum_array_copy, np.apply_along_axis(self.linear_relu, 0, sum_array)
        elif activation_function == 'softmax':
            activations = self.linear_softmax(sum_array)
            return sum_array_copy, activations
        elif activation_function == 'sigmoid':
            activations = self.linear_sigmoid(sum_array)
            return sum_array_copy, activations
        return sum_array_copy, sum_array_copy.transpose()


class Layers:
    """
    Class that bundles each individual layer of the model
    """
    def __init__(self):
        """
        Init function that sets up the layers, weights, weight updates, and layer counter to default values
        """
        self.layers = []
        self.weights = []
        self.weight_updates = []
        self.layers_counter = 0

    def add_layer(self, layer):
        """
        Adds a layer to self.layers with randomly initialized weights
        :param layer: The layer object to be added
        :return: None
        """
        layer.set_index(self.layers_counter)
        self.layers.append(layer)
        if self.layers_counter > 0:
            previous_layer = self.layers[self.layers_counter - 1].layer
            current_layer = self.layers[self.layers_counter].layer
            initialized_weights = np.random.sample((previous_layer.shape[0], current_layer.shape[0])) / 10
            self.weights.append(initialized_weights)
        self.layers_counter += 1

    @staticmethod
    def relu_gradient(relu_arr):
        """
        Returns the relu activation for the sum of the array
        :param relu_arr: The array to be activated
        :return: The output of the relu function
        """
        if relu_arr.sum() > 0:
            return 1
        else:
            return 0

    def activate_layers(self):
        """
        Activates each layer within self.layers
        :return: None
        """
        for layer_index in range(0, len(self.layers) - 1, 1):
            sum_array, activation_array = self.layers[layer_index].activate_neurons(self.weights[layer_index],
                                                                                    self.layers[
                                                                                        layer_index + 1].activation)
            if activation_array.ndim == 1:
                activation_array = np.expand_dims(activation_array, axis=1)
            sum_array = sum_array.transpose()
            self.layers[layer_index + 1].layer = activation_array
            self.layers[layer_index + 1].sum = sum_array

    @staticmethod
    def get_mae_training(mae_arr):
        """
        Calculates the mean absolute error gradient for training
        :param mae_arr: The array containing the training values
        :return: The MAE gradient
        """
        if mae_arr[0] > 0:
            return 1
        elif mae_arr[0] == 0:
            return 0
        else:
            return -1

    def back_propagate(self, cost_function, y, input_layer_obj, output_layer_obj, old_delta):
        """
        Performs back-propagation; calculates the gradient and stores them in self.weight_updates
        :param cost_function: L, the cost function used
        :param y: The labels for the training data
        :param input_layer_obj: The current input layer
        :param output_layer_obj: The current output layer
        :param old_delta: The delta values of the previous back-propagation, if applicable
        :return: Delta for the current back-propagation
        """
        cost_gradient = 1
        delta = 1
        output_layer = output_layer_obj.layer
        input_layer = input_layer_obj.layer
        output_layer_sum = output_layer_obj.sum
        activation_function = output_layer_obj.activation
        output_layer_index = output_layer_obj.index
        if np.ndim(y) == 1:
            y = np.expand_dims(y, axis=1)
        if output_layer_index == (self.layers_counter - 1):
            if cost_function == "cross entropy" or cost_function == 'binary cross entropy':
                if activation_function == 'softmax':
                    y_actual = np.zeros((output_layer.shape[0], 1))
                    y_actual[y] = 1
                    delta = y_actual - output_layer
                    updates = np.matmul(input_layer, delta.transpose())
                    self.weight_updates.append(updates)
                if activation_function == 'sigmoid':
                    delta = -output_layer * (y - output_layer)
                    updates = np.matmul(input_layer, delta.transpose())
                    self.weight_updates.append(updates)
            elif cost_function == "mse" or cost_function == "mae":
                if cost_function == "mse":
                    cost_gradient = -2 * np.subtract(output_layer, y)
                elif cost_function == "mae":
                    cost_gradient = np.subtract(output_layer, y)
                    cost_gradient = -1 * np.apply_along_axis(self.get_mae_training, 1, cost_gradient)
                    cost_gradient = np.expand_dims(cost_gradient, 1)
                if activation_function == "relu":
                    relu_gradient = output_layer_sum.copy()
                    relu_gradient = np.apply_along_axis(self.relu_gradient, 1, relu_gradient)
                    relu_gradient = np.expand_dims(relu_gradient, axis=1)
                    delta = np.multiply(relu_gradient, cost_gradient)
                    update = np.matmul(input_layer, delta.transpose())
                    self.weight_updates.append(update)
                elif activation_function is None:
                    delta = cost_gradient
                    update = np.matmul(input_layer, delta.transpose())
                    self.weight_updates.append(update)
        else:
            if activation_function == 'relu':
                relu_gradient = output_layer_sum.copy()
                relu_gradient = np.apply_along_axis(self.relu_gradient, 1, relu_gradient)
                relu_gradient = np.expand_dims(relu_gradient, axis=1)
                w_d = np.matmul(self.weights[output_layer_obj.index], old_delta)
                delta = np.multiply(w_d, relu_gradient).transpose()
                update = np.matmul(input_layer, delta)
                delta = delta.transpose()
                self.weight_updates.append(update)
            elif activation_function is None:
                delta = np.matmul(self.weights[output_layer_obj.index], old_delta).transpose()
                update = np.matmul(input_layer, delta)
                delta = delta.transpose()
                self.weight_updates.append(update)
        return delta


def plot_image(image_array):
    """
    Plots an image via matplotlib
    :param image_array: Numpy array containing the pixel values
    :return: None
    """
    local_image_array = image_array.copy()
    local_image_array = local_image_array.reshape((28, 28))
    plt.imshow(local_image_array)
    plt.show()


class Model:
    """
    Class representing the neural network model
    """
    def __init__(self, cost_function, layers):
        """
        Init function that initializes the layers
        :param cost_function:
        :param layers:
        """
        if layers is None:
            layers = []
        self.layers = Layers()
        self.output_activation = ""
        self.cost_function = cost_function
        if len(layers) > 0:
            self.add_layers(layers)
            self.output_activation = layers[-1].activation

    def initialize_moment(self):
        """
        Initializes the moment numpy arrays if using hte Adam optimizer
        :return:
        """
        moments = []
        for l_index in range(0, len(self.layers.layers), 1):
            current_layer = self.layers.layers[l_index].layer
            moment_layer = np.zeros((current_layer.shape[0], 1))
            moments.append(moment_layer)
        return moments

    def fit(self, train_x_arg, train_y_arg, test_x_arg, test_y_arg, epochs_arg, initial_lr, m):
        """
        Fits the data to the model and starts the training process
        :param train_x_arg: training features
        :param train_y_arg: training labels
        :param test_x_arg: test labels
        :param test_y_arg: test labels
        :param epochs_arg: array containing epochs
        :param initial_lr: initial learning rate
        :param m: momentum value
        :return: Training error, test error, and the x plots for graping them
        """
        lr = initial_lr
        moment_beta = m
        training_error = []
        test_error = []
        graph_x = []
        break_counter = 0
        running_averages = []
        previous_weights = []
        previous_average = 0
        total_iterations = (len(epochs_arg) * len(train_x_arg))
        if len(self.layers.layers) > 0:
            if len(train_x_arg) == len(train_y_arg):
                for e in epochs_arg:
                    for x_range in range(0, len(train_x_arg), 1):
                        current_iterations = (total_iterations - (e * len(train_x_arg) + x_range)) / total_iterations
                        moment_beta = m * (current_iterations / (1 - moment_beta + (moment_beta * current_iterations)))
                        x = train_x_arg[x_range]
                        y = train_y_arg[x_range]
                        self.layers.layers[0] = Layer(x.shape[0], None, x[:], 0)
                        self.layers.activate_layers()
                        if e == 0 and x_range == 0 and 0 > 1:
                            training_ratio = 1 - self.validate(train_x_arg, train_y_arg)
                            training_error.append(training_ratio)
                            test_ratio = 1 - self.validate(test_x_arg, test_y_arg)
                            test_error.append(test_ratio)
                            graph_x.append(e)
                        delta = None
                        for z in range(len(self.layers.layers) - 1, 0, -1):
                            input_layer_index = z - 1
                            output_layer_index = z
                            output_layer = self.layers.layers[output_layer_index]
                            input_layer = self.layers.layers[input_layer_index]
                            delta = self.layers.back_propagate(self.cost_function, y,
                                                               input_layer, output_layer, delta)
                        weight_size = len(self.layers.weights)
                        for layer_index in range(0, weight_size, 1):
                            update_index = weight_size - layer_index - 1
                            if len(previous_weights) > 0 and m is not None and m != 0:
                                updated_weights = (previous_weights[update_index] + (
                                        self.layers.weight_updates[update_index] * moment_beta)) * lr
                            else:
                                updated_weights = self.layers.weight_updates[update_index] * lr

                            self.layers.weights[layer_index] += updated_weights
                        previous_weights = self.layers.weight_updates.copy()
                        self.layers.weight_updates = []
                    test_ratio = self.validate(test_x_arg, test_y_arg)
                    running_averages.append(test_ratio)
                    if len(running_averages) > 5:
                        running_averages.pop(0)
                    if (e + 1) % 5 == 0:
                        training_ratio = self.validate(train_x_arg, train_y_arg)
                        training_error.append(training_ratio)
                        test_error.append(test_ratio)
                        graph_x.append(e)
                        current_average = 0
                        for average in running_averages:
                            current_average += average
                        current_average = current_average / len(running_averages)
                        if previous_average == 0:
                            previous_average = current_average
                        else:
                            average_ratio = current_average / previous_average
                            previous_average = current_average
                            if average_ratio > .96 and break_counter > 5:
                                break
                            elif average_ratio > .96:
                                break_counter += 1
                            else:
                                break_counter = 0
            else:
                print("Error: Incorrect input/output shape")
        return training_error, test_error, graph_x

    def get_loss(self, y):
        """
        Calculates the loss as determined by self.cost_function
        :param y: Labels for the data
        :return: The output of the loss equation
        """
        loss = 0
        if self.cost_function == "cross entropy" or self.cost_function == 'binary cross entropy':
            output_layer = self.layers.layers[-1].layer.copy()
            output_layer = np.log10(output_layer)
            actual_output = np.zeros((output_layer.shape[0], 1))
            actual_output[y] = 1
            loss = (np.multiply(-1 * output_layer, actual_output)).sum()
        elif self.cost_function == "mse":
            y_copy = np.expand_dims(y, axis=1)
            output_layer = self.layers.layers[-1].layer.copy()
            loss = np.subtract(y_copy, output_layer)
            loss = np.square(loss).sum()
        elif self.cost_function == "mae":
            y_copy = np.expand_dims(y, axis=1)
            output_layer = self.layers.layers[-1].layer.copy()
            loss = np.subtract(y_copy, output_layer)
            loss = np.abs(loss).sum()
        return loss

    def cross_entropy_validation(self, x_arg, y_arg, do_confusion_matrix=False):
        """
        Performs validation via cross-entropy
        :param x_arg: features
        :param y_arg: labels
        :param do_confusion_matrix: Boolean determining whether a confusion matrix is printed
        :return: Ratio indicating the amount correct
        """
        correct = 0
        confusion_matrix = {}
        matrix_row = {}
        for output_index in range(0, len(self.layers.layers[-1].layer), 1):
            matrix_row[output_index] = 0
        for output_index in range(0, len(self.layers.layers[-1].layer), 1):
            confusion_matrix[output_index] = matrix_row.copy()

        for z in range(0, len(x_arg), 1):
            x = x_arg[z]
            y = y_arg[z]
            self.layers.layers[0] = Layer(x.shape[0], inputs=x[:])
            self.layers.activate_layers()
            max_index = -1
            max_output = 0
            for output_index in range(0, len(self.layers.layers[-1].layer), 1):
                output_neuron = self.layers.layers[-1].layer[output_index]
                if output_neuron > max_output:
                    max_output = output_neuron
                    max_index = output_index
            if max_index == y:
                correct += 1
            if do_confusion_matrix:
                confusion_matrix[y][max_index] += 1
        correct_ratio = round(correct / len(x_arg), 2)
        if do_confusion_matrix:
            print(confusion_matrix)
            return correct_ratio
        else:
            return correct_ratio

    def binary_cross_entropy_validation(self, x_arg, y_arg, do_confusion_matrix=False):
        """
        Performs binary cross-entropy for validation
        :param x_arg: features
        :param y_arg: labels
        :param do_confusion_matrix: Determines whether a confusion matrix of the results is printed
        :return: Fraction indicating the amount correct
        """
        correct = 0
        confusion_matrix = {}
        matrix_row = {}
        for output_index in range(0, len(self.layers.layers[-1].layer), 1):
            matrix_row[output_index] = 0
        for output_index in range(0, len(self.layers.layers[-1].layer), 1):
            confusion_matrix[output_index] = matrix_row.copy()
        for z in range(0, len(x_arg), 1):
            x = x_arg[z]
            y = y_arg[z]
            self.layers.layers[0] = Layer(x.shape[0], inputs=x[:])
            self.layers.activate_layers()
            if int(self.layers.layers[-1].layer) == y:
                correct += 1
        correct_ratio = round(correct / len(x_arg), 2)
        if do_confusion_matrix:
            print(confusion_matrix)
            return correct_ratio
        else:
            return correct_ratio

    def mse_validation(self, x_arg, y_arg):
        """
        Performs validation using mean squared error for loss
        :param x_arg: features
        :param y_arg: labels
        :return: Value indicating the average MSE loss
        """
        loss = 0
        for z in range(0, len(x_arg), 1):
            x = x_arg[z]
            y = y_arg[z]
            self.layers.layers[0] = Layer(x.shape[0], None, x[:], 0)
            self.layers.activate_layers()
            loss += self.get_loss(y)
        return round(loss / len(x_arg), 2)

    def validate(self, x_arg, y_arg):
        """
        Performs validation as indicated by the loss function for the model
        :param x_arg: features
        :param y_arg: labels
        :return: loss value for the validation
        """
        if self.cost_function == "cross entropy":
            return self.cross_entropy_validation(x_arg, y_arg, True)
        elif self.cost_function == "mse" or self.cost_function == "mae":
            return self.mse_validation(x_arg, y_arg)
        elif self.cost_function == "binary cross entropy":
            return self.binary_cross_entropy_validation(x_arg, y_arg, False)

    def get_shape(self):
        """
        Returns the shape of the model
        :return: array indicating the node size for each layer
        """
        shape = []
        for layer in self.layers.layers:
            shape.append(len(layer))
        return shape

    def add_layers(self, layers):
        """
        Adds multiple layers to the model
        :param layers: list of layers to add
        :return: None
        """
        for layer in layers:
            self.layers.add_layer(layer)
        self.output_activation = layers[-1].activation

    def add_layer(self, layer):
        """
        Adds a layer to the model
        :param layer: layer object to add
        :return: None
        """
        self.layers.add_layer(layer)
        self.output_activation = layer.activation


def dataframe_to_list(data_arg):
    """
    Converts a dataframe that contains features and labels to a list, with the labels indicated by the last column
    :param data_arg: the dataframe to convert
    :return: an list containing the features and a list containing the labels
    """
    x_pd = data_arg.iloc[:, :-1]
    local_y = list(data_arg.iloc[:, -1])
    local_x = []
    for r in range(0, x_pd.shape[0]):
        local_x.append(list(x_pd.iloc[r, :]))
    return local_x, local_y


def dataframe_to_single_list(x_pd):
    """
    Converts a dataframe to a list
    :param x_pd: the dataframe to convert
    :return: a list containing the dataframe data
    """
    local_x = []
    for r in range(0, x_pd.shape[0]):
        local_x.append(x_pd[r, :])
    return local_x


def split_data(whole_data, ratio):
    """
    Splits the digit data into equal ratios for each given digit for the training and test data
    :param whole_data: The digit data to be split
    :param ratio: The train/test ratio
    :return: Dataframes containing the training and test data
    """
    raw_zero = whole_data[whole_data.iloc[:, -1] == 0]
    raw_one = whole_data[whole_data.iloc[:, -1] == 1]
    raw_two = whole_data[whole_data.iloc[:, -1] == 2]
    raw_three = whole_data[whole_data.iloc[:, -1] == 3]
    raw_four = whole_data[whole_data.iloc[:, -1] == 4]
    raw_five = whole_data[whole_data.iloc[:, -1] == 5]
    raw_six = whole_data[whole_data.iloc[:, -1] == 6]
    raw_seven = whole_data[whole_data.iloc[:, -1] == 7]
    raw_eight = whole_data[whole_data.iloc[:, -1] == 8]
    raw_nine = whole_data[whole_data.iloc[:, -1] == 9]
    train_amount = int(round(len(raw_zero) * ratio, 0)) + 1
    train_zero = raw_zero.iloc[:train_amount, :]
    train_one = raw_one.iloc[:train_amount, :]
    train_two = raw_two.iloc[:train_amount, :]
    train_three = raw_three.iloc[:train_amount, :]
    train_four = raw_four.iloc[:train_amount, :]
    train_five = raw_five.iloc[:train_amount, :]
    train_six = raw_six.iloc[:train_amount, :]
    train_seven = raw_seven.iloc[:train_amount, :]
    train_eight = raw_eight.iloc[:train_amount, :]
    train_nine = raw_nine.iloc[:train_amount, :]

    test_zero = raw_zero.iloc[train_amount:, :]
    test_one = raw_one.iloc[train_amount:, :]
    test_two = raw_two.iloc[train_amount:, :]
    test_three = raw_three.iloc[train_amount:, :]
    test_four = raw_four.iloc[train_amount:, :]
    test_five = raw_five.iloc[train_amount:, :]
    test_six = raw_six.iloc[train_amount:, :]
    test_seven = raw_seven.iloc[train_amount:, :]
    test_eight = raw_eight.iloc[train_amount:, :]
    test_nine = raw_nine.iloc[train_amount:, :]

    train_data_arg = pd.concat([train_zero, train_one, train_two, train_three, train_four, train_five,
                                train_six, train_seven, train_eight, train_nine], axis=0)
    test_data_arg = pd.concat([test_zero, test_one, test_two, test_three, test_four, test_five, test_six, test_seven,
                               test_eight, test_nine], axis=0)
    train_data_arg = train_data_arg.sample(frac=1)
    test_data_arg = test_data_arg.sample(frac=1)
    return train_data_arg, test_data_arg


def loss_by_digit(train_arg, test_arg):
    """
    Determines the individual loss for each digit
    :param train_arg: Training data
    :param test_arg: Test data
    :return: Arrays containing the training losses, test losses, and digits calculated
    """
    digits = ["all", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_losses = []
    test_losses = []
    all_train_instances = train_arg.iloc[:, :-1]
    all_train_labels = train_arg.iloc[:, -1]
    all_test_instances = test_arg.iloc[:, :-1]
    all_test_labels = test_arg.iloc[:, -1]
    train_loss = model.validate(all_train_instances.transpose(), all_train_labels)
    test_loss = model.validate(all_test_instances.transpose(), all_test_labels)
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    for d in range(1, len(digits), 1):
        digit = digits[d]
        train_instances = train_arg[train_arg.iloc[:, -1] == digit]
        train_labels = np.array(train_instances.iloc[:, -1])
        train_instances = np.array(train_instances.iloc[:, :-1])
        test_instances = test_arg[test_arg.iloc[:, -1] == digit]
        test_labels = np.array(test_instances.iloc[:, -1])
        test_instances = np.array(test_instances.iloc[:, :-1])
        train_instances = dataframe_to_single_list(train_instances)
        test_instances = dataframe_to_single_list(test_instances)
        train_loss = model.validate(train_instances, train_labels)
        test_loss = model.validate(test_instances, test_labels)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    return train_losses, test_losses, digits


def plot_images(images_arr, rows, columns):
    """
    Plots images of the digits; used if the model performs auto-encoding
    :param images_arr: numpy array containing the pixel data
    :param rows: number of images per row
    :param columns: number of images per column
    :return: None
    """
    fig, axs = plt.subplots(rows, columns)
    images_counter = 0
    for row in range(0, rows, 1):
        for column in range(0, columns, 1):
            axs[row, column].imshow(images_arr[images_counter])
            images_counter += 1
    plt.show()


# Sample implementation
learning_rate = .001
momentum_coefficient = .5
epoch_amount = 15
epoch_counter = 0
epochs = []
split_ratio = .8
while epoch_counter < epoch_amount:
    epochs.append(epoch_counter)
    epoch_counter += 1
train_data_raw = pd.read_csv("train_all_digits.csv", header=None)
test_data_raw = pd.read_csv("test_all_digits.csv", header=None)
raw_data = pd.concat([train_data_raw, test_data_raw])
train_data, test_data = split_data(raw_data, split_ratio)
train_x, train_y = dataframe_to_list(train_data)
test_x, test_y = dataframe_to_list(test_data)
train_x = np.array(train_x)
train_y = np.array(train_y)
test_x = np.array(test_x)
test_y = np.array(test_y)

all_x = np.concatenate([train_x, test_x], axis=0)
all_y = np.concatenate([train_y, test_y], axis=0)

concat_data = raw_data.sample(frac=1)
x_sampled = concat_data.iloc[:, :-1]
y_sampled = concat_data.iloc[:, -1]
# Create the model layers
model_layers = [
    Layer(x_sampled.shape[1]),
    Layer(160, "relu"),
    Layer(10, 'softmax')
]
# Create and fit the model
model = Model("cross entropy", layers=model_layers)
correct_train, correct_test, x_plots = model.fit(train_x, train_y, test_x, test_y, epochs, learning_rate,
                                                 momentum_coefficient)
# Plot the training and test loss
plt.plot(x_plots, correct_train, label="Training", color="green")
plt.plot(x_plots, correct_test, label="Testing", color="blue")
plt.ylabel("% Correct")
plt.xlabel("Epochs")
plt.legend(loc="upper right")
plt.title("% Correct Per Epoch")
plt.show()
