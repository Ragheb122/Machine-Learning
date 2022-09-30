#Ragheb Ghazi 314892506
import numpy as np
import sys
from sklearn import metrics


# soft max function
def soft_max(x):
    x_exp = np.exp(x - np.max(x))
    sum = np.sum(x_exp)
    return x_exp / sum


# normalization function return normalized array
def normalize(arr):
    normalized = arr / np.max(arr)
    return normalized


# sigmoid
sigmoid = lambda x: 1 / (1 + np.exp(-x))


# initializing the X and Y data
def initialize_data():
    # load train_x and train_y
    train_x = np.loadtxt(sys.argv[1])
    train_y = np.loadtxt(sys.argv[2])
    train_x = normalize(train_x)
    train_y = np.array(train_y)
    return {'Train_x': train_x, 'Train_y': train_y}


# initializing the w1,w2,b1,b2 parameters by giving random values
def initialize_parameters():
    train_x = np.loadtxt(sys.argv[1])
    row_num = len(train_x[0])
    w1 = np.random.randn(row_num, 256)
    b1 = np.random.randn(1, 256)
    w2 = np.random.randn(256, 10)
    b2 = np.random.randn(1, 10)
    return {'W1': w1, 'W2': w2, 'b1': b1, 'b2': b2}


# forward propagation function and (train)
def fprop_train(parameters, data, l_rate, epochs):
    w1, w2, b1, b2 = [parameters[key] for key in ('W1', 'W2', 'b1', 'b2')]
    train_x, train_y = [data[key] for key in ('Train_x', 'Train_y')]
    # shuffling the training set and the labels
    shuffleData = list(zip(train_x, train_y))
    np.random.shuffle(shuffleData)
    train_x, train_y = zip(*shuffleData)
    # Run 10 epoch
    for i in range(epochs):
        for x, y in zip(train_x, train_y):
            # hidden layer :
            z1 = np.dot(x, w1) + b1
            h1 = sigmoid(z1)
            h1 = normalize(h1)
            z2 = np.dot(h1, w2) + b2
            h2 = soft_max(z2)
            # loss :
            loss = -(np.log(h2[0][int(y)]))
            results = {'x': x, 'y': int(y), 'z1': z1, 'z2': z2, 'h1': h1, 'h2': h2, 'loss': loss, 'W2': w2}
            # updating w1,w2,b1,b2 :
            parameters = bprop(results)
            _W1, _W2, _b1, _b2 = [parameters[key] for key in ('W1', 'W2', 'b1', 'b2')]
            w1 -= (l_rate * _W1)
            b1 -= (l_rate * _b1)
            w2 -= (l_rate * _W2)
            b2 = b2 - (l_rate * _b2)
    return {'W1': w1, 'W2': w2, 'b1': b1, 'b2': b2}


# back propagation function
def bprop(params):
    x, y, z1, z2, h1, h2, loss = [params[key] for key in ('x', 'y', 'z1', 'z2', 'h1', 'h2', 'loss')]
    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]                      # for numbers [0...9]
    numbers[y] = 1
    numbers_t = np.array(np.transpose(np.matrix(h1)))             # h1.T array
    w2_t = np.array(np.transpose(np.matrix(params['W2'])))        # W2.T array
    x_t = np.array(np.transpose(np.matrix(x)))                    # x.T array
    dz2 = h2 - numbers                                            # dL/dz2
    dw2 = np.dot(numbers_t, dz2)                                  # dL/dz2 * dz2/dw2
    db2 = dz2                                                     # dL/dz2 * dz2/db2
    dz1 = (np.dot(db2, w2_t)) * sigmoid(z1) * (1 - sigmoid(z1))   # dL/dz2 * dz2/dh1 * dh1/dz1
    dw1 = np.dot(x_t, dz1)                                        # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1                                                     # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dw1, 'b2': db2, 'W2': dw2}


# predict results and write it to file
def predictions(final):
    test_file = np.array(np.loadtxt(sys.argv[3]))
    test_file = normalize(test_file)
    w1, w2, b1, b2 = [final[key] for key in ('W1', 'W2', 'b1', 'b2')]
    output = open("test_y", "w")
    # pred = []
    for x in zip(test_file):
        # calculate the hidden layer
        z1 = np.dot(x, w1) + b1
        h1 = sigmoid(z1)
        h1 = normalize(h1)
        z2 = np.dot(h1, w2) + b2
        h2 = soft_max(z2)
        # write the prediction to test_y
        y_hat = np.argmax(h2)
        #pred.append(y_hat)
        # print(y_hat)
        output.write(str(y_hat))
        output.write("\n")
    #return pred


def main():
    parameters = initialize_parameters()
    data = initialize_data()
    final = fprop_train(parameters, data, 0.1, 10)
    # train_y = np.loadtxt(sys.argv[2])
    # train_y = np.array(train_y)
    predictions(final)
    # train = np.array(np.loadtxt(sys.argv[5]))
    # print(metrics.accuracy_score(pred, train_y))


if __name__ == '__main__':
    main()
