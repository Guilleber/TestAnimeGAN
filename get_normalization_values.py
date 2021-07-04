import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

import sys


dataset_path = sys.argv[1]

transform = T.Compose([
    T.ToTensor()
])

dataset = ImageFolder(dataset_path, transform=transform)

means = [item[0].mean(dim=2).mean(dim=1) for item in dataset]
sigmas = [item[0].std(dim=2).std(dim=1) for item in dataset]
means = torch.stack(means).mean(dim=0)
sigmas = torch.stack(sigmas).mean(dim=0)
print("mean: {}".format(means))
print("std: {}".format(sigmas))
