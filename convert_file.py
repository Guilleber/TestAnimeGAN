import sys
from tqdm import tqdm

import pytorch_lightning as pl

import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.utils import save_image

from model import AnimeGAN, collate_fn


original_transforms = T.Compose([
        T.RandomResizedCrop(256, scale=(0.3, 1.0), ratio=(1.0, 1.0)),
        T.ToTensor()
    ])
original_dataset = ImageFolder(sys.argv[2], transform=original_transforms)
dataloader = DataLoader(original_dataset, batch_size=1, collate_fn=collate_fn, shuffle=False)

model = AnimeGAN.load_from_checkpoint(sys.argv[1])

for i, batch in tqdm(enumerate(list(dataloader)[200:220])):
    out_imgs = model.generator(batch)
    for j, img in enumerate(out_imgs):
        save_image(img, "{}{}_{}.jpg".format(sys.argv[3], i, j), normalize=True, value_range=(-1., 1.))
