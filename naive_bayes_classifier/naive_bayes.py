import pandas as pd


def get_data_as_bins(arg_dataset, bin_amount):
    """
    Groups continuous variables into bins
    :param arg_dataset: The dataset to make discrete
    :param bin_amount: the number of bins
    :return: Discrete version of the dataset
    """
    dataset = arg_dataset.copy()
    local_bin_dataset = pd.DataFrame(index=dataset.index, columns=dataset.columns)
    for local_column in dataset.columns[:-1]:
        local_min = dataset[local_column].min()
        local_max = dataset[local_column].max()
        increment = (local_max - local_min) / bin_amount
        column_set = pd.DataFrame(columns=[local_column])
        for v in range(0, bin_amount):
            lower = round(local_min + (v * increment), 2)
            upper = round(local_min + ((v + 1) * increment), 2)
            middle = round(lower + ((upper - lower) / 2), 0)
            dataset[local_column] = dataset[local_column].apply(
                lambda cell: middle if upper >= cell >= lower else cell)
            temp_set = dataset[dataset[local_column] == middle][local_column]
            column_set = pd.DataFrame(pd.concat([column_set[local_column], temp_set], axis=0), columns=[local_column])
            dataset[local_column] = dataset[local_column].apply(lambda cell: -1 if cell == middle else cell)
        local_bin_dataset[local_column] = column_set[local_column]
    local_bin_dataset['species'] = dataset['species']
    return local_bin_dataset


# Returns the bayesian dictionary for predictions
def get_naive_bayes_dictionary(dataset_arg, target_column):
    """
    Creates a naive bayesian tree modeled through nested dictionaries of the label, column, and value
    :param dataset_arg: The dataset to be modeled from
    :param target_column: The label column
    :return: a naive bayesian dictionary and dictionary containing the ratios of individual labels per total labels
    """
    labels = dataset_arg[target_column].unique()
    feature_columns = list(dataset_arg.columns[:-1])
    bayes_dict = {}
    ratio_label_to_total = {}
    for label_value in labels:
        ratio_label_to_total[label_value] = round(dataset_arg[dataset_arg[target_column] == label_value].shape[0] /
                                                  dataset_arg[target_column].shape[0], 2)
    for label_value in labels:
        bayes_dict[label_value] = {}
        for local_column in feature_columns:
            bayes_dict[label_value][local_column] = {}
            total = dataset_arg[local_column].shape[0]
            total_label = dataset_arg[dataset_arg[target_column] == label_value][local_column].shape[0]
            values = pd.unique(dataset_arg[local_column])
            for feature_value in values:
                value_dataset = dataset_arg[dataset_arg[local_column] == feature_value]
                value_total = value_dataset.shape[0]
                ratio_value_to_total = value_total / total
                if ratio_value_to_total != 0:
                    label_value_total = value_dataset[value_dataset[target_column] == label_value].shape[0]
                    value_given_label = label_value_total / total_label
                    probability = round(
                        value_given_label * ratio_label_to_total[label_value] / ratio_value_to_total,
                        2)
                    bayes_dict[label_value][local_column][feature_value] = probability
    return bayes_dict, ratio_label_to_total


def get_nearest_neighbor_probability(dataset, feature):
    """
    Gets the probability of a label given a column from a feature's nearest neighbor.
    :param dataset: Final bayesian nested dictionary containing all stored values for a particular label and column
    :param feature: the feature value
    :return: The probability of the label given the column from the nearest neighbor
    """
    distance = 0
    neighbor = ''
    for local_key in dataset.keys():
        local_distance = abs(local_key - feature)
        if distance == 0 or distance > local_distance:
            neighbor = local_key
            distance = local_distance
    local_probability = dataset[neighbor]
    return local_probability


def get_likeliest(bayes_dict, label_dict, instance, target_labels_arg):
    """
    Gets the likeliest label for a given example
    :param bayes_dict: The bayesian dictionary used
    :param label_dict: The label dictionary associated with bayes_dict
    :param instance: The example to derive from
    :param target_labels_arg: The possible labels to infer from
    :return: The likeliest label from target_labels
    """
    likely_label = ''
    best_probability = 0
    for target_value in target_labels_arg:
        probability = 1
        for local_column in instance.keys():
            if instance[local_column] in bayes_dict[target_value][local_column].keys():
                probability = probability * bayes_dict[target_value][local_column][instance[local_column]]
            else:
                neighbor_probability = get_nearest_neighbor_probability(bayes_dict[target_value][local_column],
                                                                        instance[local_column])
                probability = probability + neighbor_probability
        probability = round(probability * label_dict[target_value], 2)
        if probability >= best_probability:
            best_probability = probability
            likely_label = target_value
    return likely_label


def get_metrics(examples, labels, bayes_dict, label_dict):
    """
    Measures performance as a function of accuracy and confusion table values
    :param examples: Feature data
    :param labels: Labels
    :param bayes_dict: The bayesian dictionary to analyze
    :param label_dict: Associated label dictionary
    :return: The accuracy, true positive, true negative, false positive, and false negative values
    """
    unique_labels = labels.unique()
    positive_target = unique_labels[0]
    negative_target = unique_labels[1]
    local_t_p, local_f_p, local_t_n, local_f_n = 0, 0, 0, 0
    for index in range(0, examples.shape[0]):
        example = examples.iloc[index, :]
        true_label = labels.iloc[index]
        prediction = get_likeliest(bayes_dict, label_dict, example, unique_labels)
        if true_label == prediction and prediction == positive_target:
            local_t_p = local_t_p + 1
        elif true_label == negative_target and prediction == positive_target:
            local_f_p = local_f_p + 1
        elif true_label == prediction and prediction == negative_target:
            local_t_n = local_t_n + 1
        elif true_label == positive_target and prediction == negative_target:
            local_f_n = local_f_n + 1
    acc = round((local_t_p + local_t_n) / (local_t_p + local_t_n + local_f_p + local_f_n), 2)
    return acc, local_t_p, local_t_n, local_f_p, local_f_n


def train_test_split(dataset_arg, label_column, train_ratio):
    """
    Splits a dataset with two labels into training and test datasets
    :param dataset_arg: The dataset to split
    :param label_column: The column indicating the labels
    :param train_ratio: The ratio of training examples to total examples
    :return: a dataframe containing the training data and a dataframe containing the test data
    """
    train_data = pd.DataFrame(columns=dataset_arg.columns)
    test_data = pd.DataFrame(columns=dataset_arg.columns)
    labels = dataset_arg[label_column].unique().tolist()
    for label_value in labels:
        label_dataset = dataset_arg[dataset_arg[label_column] == label_value]
        label_test_index = int(label_dataset.shape[0] * train_ratio)
        train_data = pd.concat([train_data, label_dataset.iloc[:label_test_index, :]])
        test_data = pd.concat([test_data, label_dataset.iloc[label_test_index:, :]])
    train_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)
    train_data.pop('index')
    test_data.pop('index')
    return train_data, test_data


# Sample implementation using iris dataset
bins = 5
split_ratio = .8
combined_label = "versicolor or virginica"
virginica_label = "virginica"
versicolor_label = "versicolor"
setosa_label = "setosa"
target = 'species'
iris_data = pd.read_csv("iris.csv")
iris_data[target] = iris_data[target].apply(
    lambda lambda_x: combined_label if lambda_x == virginica_label or lambda_x == versicolor_label else setosa_label)
iris_data = iris_data.sample(frac=1)
target_labels = pd.unique(iris_data[target])

bin_dataset = get_data_as_bins(iris_data, bins)
training_data, testing_data = train_test_split(bin_dataset, target, split_ratio)
bayes_dictionary, label_dictionary = get_naive_bayes_dictionary(training_data, target)

x_train = training_data.iloc[:, :-1]
x_test = testing_data.iloc[:, :-1]
y_train = training_data.iloc[:, -1]
y_test = testing_data.iloc[:, -1]
train_acc, train_t_p, train_t_n, train_f_p, train_f_n = get_metrics(x_train, y_train, bayes_dictionary,
                                                                    label_dictionary)
test_acc, test_t_p, test_t_n, test_f_p, test_f_n = get_metrics(x_test, y_test, bayes_dictionary, label_dictionary)

print(train_t_p, train_t_n, train_f_p, train_f_n)
print(test_t_p, test_t_n, test_f_p, test_f_n)
print(train_acc, test_acc)
