import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

REBUILD_DATA = True

class DogsVSCats():
    IMG_SIZE = 50
    trainImages = "dogs-vs-cats\\train"
    testImages = "dogs-vs-cats\\test1"
    LABELS = {"cat" : 0, "dog" : 1}
    trainingData = []
    testingData = []
    catCount = 0
    dogCount = 0
    
    def makeTrainingData(self):
        for f in tqdm(os.listdir(self.trainImages)):
            try:
                path = os.path.join(self.trainImages, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE,))
                label = f.split(".")[0]
                self.trainingData.append([np.array(img), np.eye(2)[self.LABELS[label]]])
                if label == "cat":
                    self.catCount += 1
                elif label == "dog":
                    self.dogCount += 1 
            except Exception as e:
                pass
        np.random.shuffle(self.trainingData)
        np.save("trainingData.npy", self.trainingData)
        print("Cats: ", self.catCount)
        print("Dogs: ", self.dogCount)
    
    def makeTestingData(self):
        for f in tqdm(os.listdir(self.testImages)):
            try:
                path = os.path.join(self.testImages, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE,))
                self.testingData.append([np.array(img)])
            except Exception as e:
                pass
        np.save("testingData.npy", self.testingData)
        
        
if REBUILD_DATA == True:
    dogsVsCats = DogsVSCats()
    dogsVsCats.makeTrainingData()
    dogsVsCats.makeTestingData()
    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        x = torch.randn(50,50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)
        self.dropout = nn.Dropout()
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512,2)
     
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
        
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)
        return F.softmax(x, dim = 1)
        
    def train(self):
        optimizer = optim.Adam(net.parameters(), lr=1e-3)
        loss_function = nn.MSELoss()
        for epoch in tqdm(range(EPOCHS)):
            for i in range(0, len(trainX), BATCH_SIZE):
                batchX = trainX[i: i + BATCH_SIZE].view(-1,1,50,50).to(device)
                batchY = trainY[i: i + BATCH_SIZE].to(device)
                net.zero_grad()
                preds = net(batchX)
                loss = loss_function(preds, batchY)
                loss.backward()
                optimizer.step()
        print(loss)
    
    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for i in tqdm(range(len(valX))):
                groundTruth = torch.argmax(valY[i])
                pred = torch.argmax(net(valX[i].view(-1,1,50,50))[0]).to(device)
                if pred == groundTruth:
                    correct += 1
                total += 1
        print("Accuracy: ", round(correct/total, 3))
    

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
net = Net().to(device)
print(net)


trainData = np.load("trainingData.npy", allow_pickle = True)
testingData = np.load("testingData.npy", allow_pickle = True)
X = torch.Tensor([i[0] for i in trainData]).view(-1,50,50).to(device)
X = X/255.0
y = torch.Tensor([i[1] for i in trainData]).to(device)

VAL_PCT = 0.1
valSize = int(len(X)*VAL_PCT)

trainX = X[:-valSize]
trainY = y[:-valSize]

valX = X[-valSize:]
valY = y[-valSize:]

BATCH_SIZE = 100
EPOCHS = 100


net.train()
net.test()
