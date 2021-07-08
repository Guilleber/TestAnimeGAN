import sys
from tqdm import tqdm

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import save_image

from model import AnimeGAN, collate_fn


transforms = T.Compose([
        T.GaussianBlur(9, sigma = (5.0, 5.0)),
        T.ToTensor()
    ])
dataset = ImageFolder(sys.argv[1], transform=transforms)
save_image(dataset[0][0], "./data/test.jpg")
