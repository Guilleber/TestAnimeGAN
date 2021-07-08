import sys
import math
from tqdm import tqdm

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import save_image

from model import AnimeGAN, collate_fn


transforms = T.Compose([
        T.ToTensor()
    ])
dataset = ImageFolder(sys.argv[1], transform=transforms)
mean = []

for img, _ in tqdm(dataset):
    mean.append(img.mean(dim=2).mean(dim=1))

print("mean: {}".format(torch.stack(mean).mean(dim=0)))
