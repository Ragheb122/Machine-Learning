import numpy as np
import sys
import torch
import torch.nn.functional as nn
from torch.utils.data import SubsetRandomSampler, TensorDataset


def main():
    batch = 64
    epoches = 10
    trainy = np.loadtxt(sys.argv[2])
    # Size to split training and validation
    Size = int(len(trainy) * 0.8)
    test_x = sys.argv[3]
    test_x = np.loadtxt(test_x) / 255                       # Normalizing
    testset = torch.from_numpy(test_x).float()
    trainset = np.array(np.loadtxt(sys.argv[1]))
    trainset = trainset / 255                               # Normalizing
    labelset = np.array(np.loadtxt(sys.argv[2]))
    train_x = torch.from_numpy(trainset[:Size, :]).float()
    valid_x = torch.from_numpy(trainset[Size:]).float()
    train_y = torch.from_numpy(labelset[:Size]).long()
    valid_y = torch.from_numpy(labelset[Size:]).long()
    train_dataset = TensorDataset(train_x, train_y)
    valid_dataset = TensorDataset(valid_x, valid_y)

    # load train and validation sets as torch using torch DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch, shuffle=True)

    learning_rate = 0.1
    modelA = Model_A(image_size=28 * 28)
    optimizer = torch.optim.SGD(modelA.parameters(), lr=learning_rate)
    # training the model 10 times
    for i in range(epoches):
        train(modelA, train_loader, optimizer)
        validate(modelA, validation_loader)
        # validate(modelA, validation_loader)

    # learning_rate = 0.001
    # modelB = Model_B(image_size=28 * 28)
    # optimizer = torch.optim.Adam(modelB.parameters(), lr=learning_rate)
    #
    # for i in range(epoches):
    #     train(modelB, train_loader, optimizer)
        # validate(modelB, validation_loader)

    # learning_rate = 0.001
    # modelC = Model_C(image_size=28 * 28)
    # optimizer = torch.optim.Adam(modelC.parameters(), lr=learning_rate)

    # for i in range(epoches):
    #     print("C, epoch: " + str(i))
    #     train(modelC, train_loader, optimizer)
    #     validate(modelC, validation_loader)

    # learning_rate = 0.001
    # modelD = Model_D(image_size=28 * 28)
    # optimizer = torch.optim.Adam(modelD.parameters(), lr=learning_rate)
    #
    # for i in range(epoches):
    #     print("D, epoch: " + str(i))
    #     train(modelD, train_loader, optimizer)
    # validate(modelD, validation_loader)
        # test(modelD, test_x_loader)
        # test(modelD, train_loader)
        # test(modelD, validation_loader)

    # learning_rate = 0.009
    # modelE = Model_E(image_size=28 * 28)
    # optimizer = torch.optim.Adam(modelE.parameters(), lr=learning_rate)
    #
    # for i in range(epoches):
    #     print("E, epoch: " + str(i))
    #     train(modelE, train_loader, optimizer)
    #     validate(modelE, validation_loader)

    # learning_rate = 0.009
    # modelF = Model_F(image_size=28 * 28)
    # optimizer = torch.optim.Adam(modelF.parameters(), lr=learning_rate)
    #
    # for i in range(epoches):
    #       train(modelF, train_loader, optimizer)
    #     validate(modelF, validation_loader)

    prediction = y_prediction(modelA, testset)
    out_fname = sys.argv[4]
    # writing the predictions to the file
    file = open(out_fname, "w")
    for y in prediction:
        file.write("%s\n" % y)
    file.close()


# train function
def train(model, train_loader, optimizer):
    model.train()
    for curr_batch, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = nn.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


# predict test_y on the test_x file
def y_prediction(model, test_x):
    model.eval()
    predictions = []
    for data in test_x:
        output = model(data)
        predict = output.max(1, keepdim=True)[1]
        _predict = str(int(predict))
        predictions.append(_predict)
    return predictions

# def test(model, test_loader):
#     model.eval()
#     test_loss = 0
#     accuracy_counter = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             output = model(data)
#             test_loss += nn.nll_loss(output, target, reduction='sum').item()
#             prediction = output.max(1, keepdim=True)[1]
#             accuracy_counter += prediction.eq(target.view_as(prediction)).cpu().sum()


    # test_loss /= len(test_loader.dataset)
    # print('\nTest Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #     test_loss, accuracy_counter, len(test_loader.dataset), 100. * accuracy_counter / len(test_loader.dataset)))


def validate(model, validation_loader):
    model.eval()
    validation_loss = 0
    accuracy_counter = 0
    with torch.no_grad():
        for data, target in validation_loader:
            output = model(data)  # forward.
            validation_loss += nn.nll_loss(output, target, reduction='sum').item()
            prediction = output.max(1, keepdim=True)[1]
            accuracy_counter += prediction.eq(target.view_as(prediction)).cpu().sum()


   # validation_loss /= (len(validation_loader) * batch)
   #  print('\nValidation Set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
   #    validation_loss, accuracy_counter, (len(validation_loader) * batch),
   #     100. * accuracy_counter / (len(validation_loader) * batch)))



# Model A
class Model_A(torch.nn.Module):
    def __init__(self, image_size):
        super(Model_A, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)


# Model B
class Model_B(torch.nn.Module):
    def __init__(self, image_size):
        super(Model_B, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)


# Model C - with Dropout
class Model_C(torch.nn.Module):
    def __init__(self, image_size):
        super(Model_C, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc2 = torch.nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        m = torch.nn.Dropout(p=0.45)
        x = m(x)
        x = nn.relu(self.fc1(x))
        m = torch.nn.Dropout(p=0.45)
        x = m(x)
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)


# Model D - with Batch Normalization
class Model_D(torch.nn.Module):
    def __init__(self, image_size):
        super(Model_D, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 100)
        self.fc_bn_0 = torch.nn.BatchNorm1d(100)
        self.fc1 = torch.nn.Linear(100, 50)
        self.fc_bn_1 = torch.nn.BatchNorm1d(50)
        self.fc2 = torch.nn.Linear(50, 10)
        self.fc_bn_2 = torch.nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc_bn_0(self.fc0(x)))
        x = nn.relu(self.fc_bn_1(self.fc1(x)))
        x = nn.relu(self.fc_bn_2(self.fc2(x)))
        return nn.log_softmax(x, dim=1)


# Model E
class Model_E(torch.nn.Module):
    def __init__(self, image_size):
        super(Model_E, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 10)
        self.fc5 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = nn.relu(self.fc0(x))
        x = nn.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)


# Model F
class Model_F(torch.nn.Module):
    def __init__(self, image_size):
        super(Model_F, self).__init__()
        self.image_size = image_size
        self.fc0 = torch.nn.Linear(image_size, 128)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 10)
        self.fc5 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return nn.log_softmax(x, dim=1)


if __name__ == "__main__":
    main()
