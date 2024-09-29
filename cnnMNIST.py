import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Definindo a arquitetura da CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Configuração do dispositivo (GPU se disponível, senão CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Preparação dos dados
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Inicialização do modelo, função de perda e otimizador
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Configuração do TensorBoard
writer = SummaryWriter('runs/mnist_experiment')

# Treinamento
num_epochs = 5
for epoch in range(num_epochs):
    running_loss = 0.0
    dataQt = 0
    for i, data in enumerate(trainloader, 0):        
        inputs, labels = data[0].to(device), data[1].to(device)
        dataQt += inputs.shape[0]
        
        image = inputs[0].cpu()
        image_np = image.squeeze().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(image_np, cmap='gray')
        plt.title(f"Dígito: {labels[0]}")
        plt.axis('off')
        plt.show()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / dataQt
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {avg_loss:.3f}')
    writer.add_scalar('training loss', avg_loss, epoch * len(trainloader) + i)    

print('Treinamento concluído!')

# Avaliação
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Acurácia da rede em 10000 imagens de teste: {accuracy}%')
writer.add_scalar('test accuracy', accuracy, num_epochs)

# Fechar o writer do TensorBoard
writer.close()