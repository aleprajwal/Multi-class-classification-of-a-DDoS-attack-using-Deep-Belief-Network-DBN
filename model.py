from numpy import loadtxt, random
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd


# load the dataset
def load_dataset():
    dataset = loadtxt('TrainData.csv', delimiter=',', skiprows=1)
    # split into input (X) and output (y) variables
    X = dataset[:, 0:8]
    y = dataset[:, 8]
    return X, y


# load DBN pretrained weight
def load_weight():
    paths = ['RBM1.csv', 'RBM2.csv', 'RBM3.csv', 'RBM4.csv']
    weights = list()
    for path in paths:
        df = pd.read_csv(path)
        df = df.drop(columns='hidden0', index=8, axis=0)  # drop bias
        np_array = df.to_numpy()
        weights.append(np_array)
    return weights


# define the keras model
model = Sequential()
model.add(Dense(25, input_dim=8, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

weight = load_weight()

# setup input and first hidden layer with pretrained weight
bias = random.rand(25)
model.layers[0].set_weights([weight[0], bias])
# setup first and second hidden layer with pretrained weight
bias = random.rand(10)
model.layers[1].set_weights([weight[1], bias])
# setup second and third hidden layer with pretrained weight
model.layers[2].set_weights([weight[2], bias])
bias = random.rand(1)
# setup third and output layer with pretrained weight
model.layers[3].set_weights([weight[3], bias])
# print(model.layers[0].get_weights()[0])

X, y = load_dataset()
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=100, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
