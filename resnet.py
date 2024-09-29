import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Carregar modelo ResNet pré-treinado
model = models.resnet50(pretrained=True)
print(model)
model.eval()  # Modo de avaliação

# Definir transformações para a imagem
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carregar e preparar a imagem
url = "dados/Cat03.jpg"
img = Image.open(url)
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)

# Fazer a predição
with torch.no_grad():
    output = model(batch_t)

# Carregar as classes do ImageNet
with open('dados\imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

# Obter as 5 principais previsões
_, indices = torch.sort(output, descending=True)
percentages = torch.nn.functional.softmax(output, dim=1)[0] * 100
for idx in indices[0][:5]:
    print(f"{classes[idx]}: {percentages[idx].item():.2f}%")