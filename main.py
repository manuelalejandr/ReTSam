!pip install pytorch_lightning
!pip install lightly

import os
import random
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import mode
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import lightning as L


def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(2023)
L.seed_everything(42)

batch_size = 64


simclr_transform = SimCLRTransform(
    input_size=128,
    cj_strength=0.5,
    gaussian_blur=0.0,
)

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

train_data = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

train_data2 = torchvision.datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=test_transform,
)

test_data = torchvision.datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=test_transform,
)

train_dataloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=False
)

train_dataloader2 = torch.utils.data.DataLoader(
    train_data2,
    batch_size=batch_size,
    shuffle=False
)

test_dataloader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False
)

def compute_ordered_neighbors(z0, z1, n_neighbors):
    # Calcula la matriz de distancias
    distances = torch.cdist(z0, z1)

    # Ordena los índices por distancia
    _, indices = torch.sort(distances, dim=1)

    return indices[:, :n_neighbors]

class SimCLR(nn.Module):
    def __init__(self, backbone, batch, n_neg_neighbors, weight_decay=1e-1):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(2048, 2048, 128)
        self.n_neg_neighbors = n_neg_neighbors
        self.batch = batch
        self.w = nn.Parameter(torch.ones(self.n_neg_neighbors-1, 1), requires_grad=True)
        self.weight_decay = weight_decay
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

weights = torchvision.models.ResNet50_Weights.DEFAULT
resnet = torchvision.models.resnet50(weights=weights)
backbone = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.AdaptiveAvgPool2d(1)
)

epochs = 200
lr_factor = batch_size / batch_size

criterion = NTXentLoss()
model = SimCLR(backbone, batch=batch_size, n_neg_neighbors=15)
optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

print("Starting Training")
for epoch in range(epochs):
    total_loss = 0
    for batch in tqdm(train_dataloader):
        x0, x1 = batch[0]
        y = batch[1].to(device)
        x0 = x0.to(device)
        x1 = x1.to(device)
        z0 = model(x0)
        z1 = model(x1)

        indices_n = compute_ordered_neighbors(z0, z1, 15)
        z_n = z_n = torch.mean(torch.stack([z1[i] for i in indices_n[:,1:]]), dim=1)

        loss = triplet_loss(z0, z1, z_n) + (0.5) * criterion(z0, z1)

        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    scheduler.step()
    avg_loss = total_loss / len(train_dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

def generate_embeddings_test(model, dataloader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for img, lab in tqdm(dataloader):
            img = img.to(device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            labels.extend(lab)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings.cpu())
    return torch.tensor(embeddings), torch.tensor(labels)

def generate_embeddings_train(model, dataloader):
    embeddings = []
    labels = []
    with torch.no_grad():
        for img, lab in tqdm(dataloader):
            img = img.to(device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            labels.extend(lab)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings.cpu())
    return torch.tensor(embeddings), torch.tensor(labels)

model.eval()
test_embedding, test_label = generate_embeddings_test(model, test_dataloader)
train_embedding, train_label = generate_embeddings_train(model, train_dataloader2)

ttest_1000 = []
for i in range(10):
    ttest_1000.append(np.random.choice(torch.where(test_label==i)[0].cpu(), 100))
test_1000 = np.concatenate(ttest_1000, 0)



np.random.seed(2023)
total = 1000

y_preds = []

maps = []


knn_acc = 0
for j in test_1000:

  distances = cdist(np.array(test_embedding[j].cpu()).reshape((1, test_embedding.shape[1])), train_embedding.cpu(), metric="euclidean")

  distances_dict = dict(zip(range(len(distances[0])), distances[0]))


  list_dist_orders = dict(sorted(distances_dict.items(), key=lambda x: x[1], reverse=False))
  dist_img = list(list_dist_orders.values())

  index_img = list(list_dist_orders.keys())[:total]


  truess = 0
  sum = 0


  for quant, i in enumerate(index_img):
    if test_label[j]==train_label[i]:
      truess += 1
      sum += truess/(quant+1)
  maps.append(sum/total)


print("MAP at 1000 without binary:", np.sum(maps)/len(maps))




        
