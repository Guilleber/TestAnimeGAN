import torch
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

import pytorch_lightning as pl

from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from modules import Generator, Discriminator
import losses


def collate_fn(batch):
    batch = torch.stack([item[0] for item in batch])
    return batch*2 - 1


class AnimeGAN(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        real = batch['real']
        gray = batch['gray']
        smooth = batch['smooth']

        input = batch['original']
        fake = self.generator(input)

        #optimize discriminator
        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake.detach())
        d_gray = self.discriminator(gray)
        d_smooth = self.discriminator(smooth)

        err_d = self.hparams.d_weight * losses.d_loss(d_real, d_fake, d_gray, d_smooth)

        d_opt.zero_grad()
        self.manual_backward(err_d)
        d_opt.step()

        #optimize generator
        d_fake = self.discriminator(fake)
        err_g = self.hparams.g_weight * losses.g_loss(d_fake)

        err_g += self.hparams.con_weight*losses.con_loss(input, fake)
        err_g += self.hparams.color_weight*losses.color_loss(input, fake)
        err_g += self.hparams.style_weight*losses.style_loss(input, fake)

        g_opt.zero_grad()
        self.manual_backward(err_g)
        g_opt.step()

        self.log_dict({'g_loss': err_g, 'd_loss': err_d})
        return

    def train_dataloader(self):
        dataloaders = {}

        real_transforms = T.Compose([
                T.ToTensor()
            ])
        real_dataset = ImageFolder(hparams.real_path, transforms=real_transforms)
        dataloaders['real'] = DataLoader(real_dataset, collate_fn=collate_fn)

        original_transforms = T.Compose([
                T.RandomResizedCrop(256, scale=(0.3, 1.0), ratio=(1.0, 1.0)),
                T.ToTensor()
            ])
        original_dataset = ImageFolder(hparams.original_path, transforms=original_transforms)
        dataloaders['original'] = DataLoader(original_dataset, collate_fn=collate_fn)

        gray_transforms = T.Compose([
                T.GrayScale(num_output_channels=3),
                T.ToTensor()
            ])
        gray_dataset = ImageFolder(hparams.real_path, transforms=gray_transforms)
        dataloaders['gray'] = DataLoader(gray_dataset, collate_fn=collate_fn)

        smooth_transforms = T.Compose([
                T.GaussianBlur(5, sigma(2.0, 5.0)),
                T.ToTensor()
            ])
        smooth_dataset = ImageFolder(hparams.real_path, transforms=smooth_transforms)
        dataloaders['smooth'] = DataLoader(smooth_dataset, collate_fn=collate_fn)

        return dataloaders

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator, self.hparams.g_lr)
        d_opt = torch.optim.Adam(self.discriminator, self.hparams.d_lr)
        return g_opt, d_opt
