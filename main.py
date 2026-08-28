import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
# The only deps we need!

data = pd.read_csv('data/train.csv')

data.head()

data = np.array(data) # converting to an array

m, n = data.shape # m is the number of rows(data lenght) and n is the number of columns(785)

# print(m, n, "dgdfdg", data.shape)

np.random.shuffle(data) # mixing our data up

# We are going to split our data in to two, one for training our model and the other for testing it
# data_train and data_test

data_test = data[0:1000].T # Transposing so we have 785 rows and 1000 columns.

label_test= data_test[0] # the labels: only first row
feature_test = data_test[1: n] # the features, 1-785. 2nd row to 785th
feature_test = feature_test/ 255.0 # normalize because value gets too large

data_train = data[1000:m].T # all the remaining data we have
label_train = data_train[0]
feature_train = data_train[1:n]
feature_train = feature_train / 255.0

# print(label_train.shape, feature_train[:,0].shape)

#Inititalize the weights and baises
def init_params (): 
    W1 = np.random.rand(10,784) - 0.5 # geenerate random numbers between 0 and 1 in the shape of one input layer, 10 rows and 784 columns which are the weights we need
    b1 = np.random.rand(10,1) - 0.5 # same thing but this time the shape of our bias
    W2 = np.random.rand(10,10) - 0.5 # 10 on hidden layer and 10 on output. 10 X 10
    b2 = np.random.rand(10,1) - 0.5 
    return W1, b1, W2, b2

# ro update the params
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2


# Our activation function for A1
def ReLU(Z):
    return np.maximum(0, Z) # less than 0 returns 0 and other returns Z

# Softmax activation
def softmax(Z):
    # stopped using this because of overflow error, exponential gets too big
    # return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

    # new one
    Z = Z - np.max(Z, axis=0, keepdims=True)
    return np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)


# One hot encoding
def one_hot_encode(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1)) # create an array the size of our labels with zeros and make it as wide as the number of outputs(10)
    one_hot_Y[np.arange(Y.size), Y] = 1 # basically just flips the items that matches our label to 1
    one_hot_Y = one_hot_Y.T # transpose so it stays consistent with our structure
    return one_hot_Y

def ReLU_derivative(Z):
    return Z > 0 # a really smart way to usd booleans to handle it..Calculus!

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

# X is input tensor
def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) +b1 # Z1 equation
    A1 = ReLU(Z1) # Activation for layer 1

    Z2 = W2.dot(A1) + b2 # second layer
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2


def backward_propagation(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot_encode(Y)
    dZ2 = A2 - one_hot_Y # one first backward prop equation
    dW2 = 1/m * dZ2.dot(A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_derivative(Z1)
    dW1 = 1/m * dZ1.dot(X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2



########
# The training
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W2, X, Y)

        # update our params 
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if i % 50 == 0:
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))

    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(feature_train, label_train, 500, 0.10)
# got 85.6 percent on my first run with this, iteration 500 and alpha as 0.
# 92.5 percent with 5000 runs and 0.10 learning rate.



## Testing

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = feature_train[:, index, None]
    prediction = make_predictions(feature_train[:, index, None], W1, b1, W2, b2)
    label = label_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

# test_prediction(8, W1, b1, W2, b2) # test with result from our gradient descent 

dev_predictions = make_predictions(feature_test, W1, b1, W2, b2)
dev_accuracy = get_accuracy(dev_predictions, label_test)
print("Test data accuracy: ", dev_accuracy) # 85.8%
