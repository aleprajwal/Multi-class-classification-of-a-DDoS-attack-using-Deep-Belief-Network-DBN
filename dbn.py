from numpy import loadtxt

import pandas as pd
import numpy as np

# load the dataset
import rbm


def load_dataset():
    dataset = loadtxt('DummyCSV/TrainData.csv', delimiter=',', skiprows=1)
    # split into input (X)
    # output (or the class) variables is ignored because the DBN is trained in an unsupervised way
    X = dataset[:, 0:8]
    return X


# hard-coded dbn (stacking 4 rbm's manually) -> due to time constraint
def train_dbn(X, visible_layer):
    """

    :param X: dataset
    :param visible_layer: input size (visible unit)
    :return: list of weights of each rbm
    """

    # layer: number of layers eg. [5, 10, 1] represent 2 rbm i.e, 3 layers(total) with 5, 10, and 1 node respectively
    layer = [visible_layer, 25, 10, 10, 1]

    rbm_1 = rbm.train_rbm(dataset=X, num_visible=layer[0], num_hidden=layer[1])
    # delete first bias column
    X = np.delete(rbm_1[1], 0, 1)  # after training value of hidden units in 1st rbm becomes input unit for next rbm
    rbm_1 = np.delete(rbm_1[0], 0, 0)  # delete bias -> first row
    rbm_1 = np.delete(rbm_1, 0, 1)  # delete bias -> first column

    rbm_2 = rbm.train_rbm(dataset=X, num_visible=layer[1], num_hidden=layer[2])
    # delete first bias column
    X = np.delete(rbm_2[1], 0, 1)  # after training value of hidden units in 1st rbm becomes input unit for next rbm
    rbm_2 = np.delete(rbm_2[0], 0, 0)  # delete bias -> first row
    rbm_2 = np.delete(rbm_2, 0, 1)  # delete bias -> first column

    rbm_3 = rbm.train_rbm(dataset=X, num_visible=layer[2], num_hidden=layer[3])
    # delete first bias column
    X = np.delete(rbm_3[1], 0, 1)  # after training value of hidden units in 1st rbm becomes input unit for next rbm
    rbm_3 = np.delete(rbm_3[0], 0, 0)  # delete bias -> first row
    rbm_3 = np.delete(rbm_3, 0, 1)  # delete bias -> first column

    rbm_4 = rbm.train_rbm(dataset=X, num_visible=layer[3], num_hidden=layer[4])
    rbm_4 = np.delete(rbm_4[0], 0, 0)  # delete bias -> first row
    rbm_4 = np.delete(rbm_4, 0, 1)  # delete bias -> first column

    weights = list()
    weights.append(rbm_1)
    weights.append(rbm_2)
    weights.append(rbm_3)
    weights.append(rbm_4)

    print("\t\t\n\n[INFO] Pre-Training completed\n\n")

    return weights


if __name__ == "__main__":
    X = load_dataset()
    print(X.shape)
    visibleUnits = X.shape[1]
    # hiddenUnits = 2
    # rbm = rbm.train_rbm(dataset=X, num_visible=visibleUnits, num_hidden=hiddenUnits)
    # print(rbm[0])
    # print(rbm[0].shape)

    # rbm_w = np.delete(rbm[0], 0, 0)
    # rbm_w = np.delete(rbm_w, 0, 1)

    # print(rbm_w, rbm_w.shape)

    dbn_model = train_dbn(X, visibleUnits)

    for weight in dbn_model:
        print(weight)
        print(weight.shape)
