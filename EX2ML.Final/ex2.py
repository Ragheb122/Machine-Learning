# Ragheb Ghazi 314892506
import sys
import numpy as np

output = 3
# loading text files
train_x = np.loadtxt(sys.argv[1], delimiter=',', dtype=float)
train_y = np.loadtxt(sys.argv[2], delimiter=',', dtype=float)
test_x = np.loadtxt(sys.argv[3], delimiter=',', dtype=float)
out_fname = sys.argv[4]
# normalization :
n_train = (train_x - train_x.mean(0)) / train_x.std(0)
n_test = (test_x - test_x.mean(0)) / test_x.std(0)
# bias column
full = np.full(len(train_x), 1)
bias = full.reshape(len(train_x), 1)

# using most common function for KNN to find the most common class and return it.
def most_Common(list1):
    counter = 0
    number = list1[0]
    for i in list1:
        current_frequency = list1.count(i)
        if current_frequency > counter:
            counter = current_frequency
            number = i
    return number


# shuffle function using np
def shuffle(set):
    np.random.shuffle(set)


class KNN:
    # initialization function
    def __init__(self, k, train_x, train_y):
        self.k = k
        self.train_x = train_x
        self.train_y = train_y

    # training function
    def train(self, x_sample):
        # compute distances
        distances = [np.sqrt(np.sum((x_sample - x_train) ** 2)) for x_train in self.train_x]
        # get k nearest samples, labels
        k_sort = np.argsort(distances)[: self.k]
        nearest_classes = [self.train_y[i] for i in k_sort]
        # majority vote, using most_Common function
        most_common = most_Common(nearest_classes)
        return most_common

    # predictions function
    def predict(self, train_x):
        predicted_classes = [self.train(x) for x in train_x]
        predictions = np.array(predicted_classes)
        return predictions


class Perceptron:
    # initialization function
    def __init__(self, train_x, train_y, learning_rate):
        self.train_x = train_x
        self.train_y = train_y
        self.learning_rate = learning_rate
        # ready for train
        self.train_x = np.append(self.train_x, bias, axis=1)
        self.length = len(self.train_x[0])

    # training function
    def train(self, epochs):
        weights = np.zeros((3, self.train_x.shape[1]))
        for epoch in range(epochs):
            to_shuffle = list(zip(self.train_x, self.train_y))
            shuffle(to_shuffle)
            for x, y in to_shuffle:
                u = self.learning_rate * x
                y_hat = np.argmax(np.dot(weights, x))
                if int(y) == int(y_hat):
                    weights[int(y)] = weights[int(y)]
                    weights[int(y_hat)] = weights[int(y_hat)]
                else:
                    weights[int(y_hat)] = weights[int(y_hat)] - u
                    weights[int(y)] = weights[int(y)] + u
        return weights

    # predictions function
    def predict(self, weights, row):
        row = np.append(row, 1)
        predictions = np.argmax(np.dot(weights, row))
        return predictions


class SVM:
    # initialization function
    def __init__(self, train_x, train_y, learning_rate, lambada):
        self.train_x = train_x
        self.train_y = train_y
        self.learning_rate = learning_rate
        self.lambada = lambada
        # ready for train
        self.train_x = np.append(self.train_x, bias, axis=1)
        self.length = len(self.train_x[0])

    # training function
    def train(self, epochs):
        weights = np.zeros((output, self.length))
        for epoch in range(epochs):
            to_shuffle = list(zip(self.train_x, self.train_y))
            shuffle(to_shuffle)
            for x, y in to_shuffle:
                y_hat = np.argmax(np.dot(weights, x))
                loss = max(0, 1 - np.dot(weights[int(y)], x) + np.dot(weights[int(y_hat)], x))
                u1 = self.lambada * self.learning_rate
                u2 = self.learning_rate * x
                if loss <= 0:
                    weights *= (1 - u1)
                else:
                    weights_length = len(weights)
                    weights[int(y_hat)] = (1 - u1) * weights[int(y_hat)] - u2
                    weights[int(y)] = (1 - u1) * weights[int(y)] + u2
                    for i in range(weights_length):
                        if i != int(y):
                            if i != int(y_hat):
                                weights[i] = weights[i] * (1 - u1)
            if epoch == 10 or epoch == 20 or epoch == 30 or epoch == 40 or epoch == 50:
                self.learning_rate *= 0.05
        return weights

    # predictions function
    def predict(self, weights, row):
        row = np.append(row, 1)
        predictions = np.argmax(np.dot(weights, row))
        return predictions


class PA:
    # initialization function
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        # ready for train
        self.train_x = np.append(self.train_x, bias, axis=1)
        self.length = len(self.train_x[0])

    # training function
    def train(self, epochs):
        weights = np.zeros((output, self.length))
        for epoch in range(epochs):
            to_shuffle = list(zip(self.train_x, self.train_y))
            shuffle(to_shuffle)
            for x, y in to_shuffle:
                y_hat = np.argmax(np.dot(weights, x))
                loss = max(0, 1 - np.dot(weights[int(y)], x) + (np.dot(weights[int(y_hat)], x)))
                n_loss = loss / (2 * (np.linalg.norm(x) ** 2))
                if int(y) == int(y_hat):
                    weights[int(y)] = weights[int(y)]
                    weights[int(y_hat)] = weights[int(y_hat)]
                else:
                    weights[int(y)] = weights[int(y)] + n_loss * x
                    weights[int(y_hat)] = weights[int(y_hat)] - n_loss * x
        return weights

    # prediction function
    def predict(self, weights, row):
        row = np.append(row, 1)
        predictions = np.argmax(np.dot(weights, row))
        return predictions


def main():
    # KNN :
    knn = KNN(5, train_x, train_y)
    KNN_pred = knn.predict(test_x)
    # Perceptron :
    perceptron = Perceptron(n_train, train_y, 0.05)
    per_weights = perceptron.train(100)
    # SVM :
    svm = SVM(n_train, train_y, 0.05, 0.1)
    svm_weights = svm.train(100)
    # PA :
    pa = PA(n_train, train_y)
    pa_weights = pa.train(100)
    # open output file to write
    out_put_file = open(out_fname, "w")
    # array for each algorithm to check accuracy
    perceptron_arr, svm_arr, pa_arr, j = [], [], [], 0
    rows = zip(n_test, test_x)
    # filling arrays of predictions for each algorithm
    for n_row, row in rows:
        perceptron_yhat = perceptron.predict(per_weights, n_row)
        perceptron_arr.append(perceptron_yhat)
        svm_yhat = svm.predict(svm_weights, n_row)
        svm_arr.append(svm_yhat)
        pa_yhat = pa.predict(pa_weights, n_row)
        pa_arr.append(pa_yhat)
    # writing to the output file
    for i in range(len(test_x)):
        out_put_file.write(f"knn: {int(KNN_pred[j])}, perceptron: {perceptron_arr[j]}, svm: {svm_arr[j]}, pa: {pa_arr[j]}\n")
        j += 1
    # checking accuracies
    #print("final SVM: ", metrics.accuracy_score(svm_arr, train_y))
    #print("final pa: ", metrics.accuracy_score(pa_arr, train_y))
    #print("final perceptron: ", metrics.accuracy_score(perceptron_arr, train_y))
    #print("final KNN: ", metrics.accuracy_score(KNN_pred, train_y))
    out_put_file.close()


if __name__ == "__main__":
    main()
