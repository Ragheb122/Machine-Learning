import torch.nn as nn
from gcommand_dataset import GCommandLoader
import torch


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fullyConn1 = nn.Linear(3840, 1024)
        self.fullyConn2 = nn.Linear(1024, 128)
        self.fullyConn3 = nn.Linear(128, 30)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)
        x = self.fullyConn1(x)
        x = self.fullyConn2(x)
        x = self.fullyConn3(x)
        return x


def train_data(train_loader, loss_fun, model, epochs):
    model.train()
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fun(outputs, labels)
            loss.backward()
            optimizer.step()



def test():
    map = data_set.class_to_idx
    map = {v: k for k, v in map.items()}
    modelPredictions = []
    model.eval()
    currentCounter = 0
    for data, labels in test_loader:
        output = model(data)
        predictions = output.max(1, keepdim=True)[1]
        for prediction in predictions.data:
            path = test_loader.sampler.data_source.spects[currentCounter][0]
            number = path.split("/")
            numb = number[len(number) - 1]
            modelPredictions.append([numb, (map[prediction.item()])])
        currentCounter += 1
    modelPredictions.sort(key=lambda x: int(x[0].split('.')[0]))
    return modelPredictions


def write_to_file(modelPredictions):
    outputFile = open("test_y", "w")
    for finalPrediction in modelPredictions:
        finalPredictionString = str(finalPrediction[0]) + "," + str(finalPrediction[1] + "\n")
        outputFile.write(finalPredictionString)
    outputFile.close()

if __name__ == "__main__":

    data_set = GCommandLoader('./data/train')
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=100, shuffle=True, pin_memory=True)

    validation_set = GCommandLoader('./data/valid')
    validLoader = torch.utils.data.DataLoader(validation_set, batch_size=100, shuffle=False, pin_memory=True)

    testDataset = GCommandLoader('./data/test')
    test_loader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False, pin_memory=True)

    model = MyNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # for the back-propagation
    loss_fun = nn.CrossEntropyLoss()
    train_data(train_loader, loss_fun, model, 10)
    modelPredictions = test()
    write_to_file(modelPredictions)

