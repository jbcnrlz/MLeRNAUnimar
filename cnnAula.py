import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CNNDeBoas(nn.Module):
    def __init__(self):
        super(CNNDeBoas,self).__init__()
        self.convPart = nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.featureExtractionPart = nn.Sequential(
            nn.Linear(64*7*7,128),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )
        self.classificador = nn.Linear(128,10)

    def forward(self,x):
        x = self.convPart(x)
        x = x.view(-1,64*7*7)
        x = self.featureExtractionPart(x)
        return self.classificador(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ])

    trainSet = torchvision.datasets.MNIST(
        root='./data', train=True,download=True,transform=ts
    )
    trainLoaded = DataLoader(trainSet,batch_size=64,shuffle=True)

    testSet = torchvision.datasets.MNIST(
        root='./data', train=False,download=True,transform=ts
    )
    testLoader = DataLoader(testSet,batch_size=64,shuffle=False)

    model = CNNDeBoas().to(device)
    lossFunction = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(),lr=0.001)

    numEpochs = 5
    for ep in range(numEpochs):
        runningLoss = 0
        dataQuant = 0
        for i, data in enumerate(trainLoaded):
            inputs = data[0].to(device)
            labels = data[1].to(device)

            dataQuant += inputs.shape[0]

            opt.zero_grad()
            outputs = model(inputs)
            loss = lossFunction(outputs,labels)
            loss.backward()
            opt.step()

            runningLoss += loss.item()

        print("Época %d ==> Loss %f" % (ep,runningLoss / dataQuant))
    print('Treinamento concluido!')

    corretos = 0
    total = 0
    with torch.no_grad():
        for data in testLoader:
            inputs = data[0].to(device)
            labels = data[1].to(device)

            outputs = model(inputs)
            _, pred = torch.max(outputs.data,1)
            total += labels.shape[0]
            corretos += (pred == labels).sum().item()
    
    print("Acurácia = %f" % (corretos / total))

if __name__ == '__main__':
    main()