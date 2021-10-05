from numpy import loadtxt, random
from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
import pandas as pd
import matplotlib.pyplot as plt

from dbn import train_dbn


# load the dataset
def load_dataset():
    dataset = loadtxt('DummyCSV/iris_shuffled.csv', delimiter=',', skiprows=1)
    # split into input (X) and output (y) variables
    X = dataset[:, 0:4]
    y = dataset[:, 4:]
    return X, y


# load DBN pretrained weight
def load_weight():
    paths = ['DummyCSV/RBM1.csv', 'DummyCSV/RBM2.csv', 'DummyCSV/RBM3.csv', 'DummyCSV/RBM4.csv']
    weights = list()
    for path in paths:
        df = pd.read_csv(path)
        df = df.drop(columns='hidden0', index=8, axis=0)  # drop bias
        np_array = df.to_numpy()
        weights.append(np_array)
    return weights


# build model
def build_model(weight):
    # define the keras model
    model = Sequential()
    model.add(Dense(25, input_dim=4, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(3, activation='softmax'))

    # setup input and first hidden layer with pretrained weight
    bias = random.rand(25)
    model.layers[0].set_weights([weight[0], bias])
    # setup first and second hidden layer with pretrained weight
    bias = random.rand(10)
    model.layers[1].set_weights([weight[1], bias])
    # setup second and third hidden layer with pretrained weight
    model.layers[2].set_weights([weight[2], bias])
    # bias = random.rand(1)
    # setup third and output layer with pretrained weight
    # model.layers[3].set_weights([weight[3], bias])
    # print(model.layers[0].get_weights()[0])

    return model


X, y = load_dataset()
weight = load_weight()
# weight = train_dbn(X, X.shape[1]) # pretrained weight

model = build_model(weight=weight)

# model summary
print(model.summary())
plot_model(model, "model.png", show_shapes=True)


# compile the keras model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
clf = model.fit(X, y, validation_split=0.33, epochs=100, batch_size=10)
model.save('ddos_model')

# list of data in history
print(clf.history.keys())

# summarize history for accuracy
plt.plot(clf.history['accuracy'])
plt.plot(clf.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(clf.history['loss'])
plt.plot(clf.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# evaluate the keras model
loss, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
print('Loss: %.2f' % loss)


