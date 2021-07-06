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
        self.save_hyperparameters(hparams)
        self.generator = Generator()
        self.discriminator = Discriminator(self.hparams.d_channels, self.hparams.d_layers)

        self.automatic_optimization = False

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Model")
        parser.add_argument('--real_path', type=str, default="./data/original/")
        parser.add_argument('--original_path', type=str, default="./data/Shinkai/")

        parser.add_argument('--d_channels', type=int, default=64)
        parser.add_argument('--d_layers', type=int, default=3)

        parser.add_argument('--init_lr', type=float, default=2e-4, help='The learning rate')
        parser.add_argument('--g_lr', type=float, default=2e-5, help='The learning rate')
        parser.add_argument('--d_lr', type=float, default=4e-5, help='The learning rate')

        parser.add_argument('--g_weight', type=float, default=300.0, help='Weight about GAN')
        parser.add_argument('--d_weight', type=float, default=300.0, help='Weight about GAN')
        parser.add_argument('--con_weight', type=float, default=1.1, help='Weight about VGG19')
        parser.add_argument('--style_weight', type=float, default=2.8, help='Weight about style')
        parser.add_argument('--color_weight', type=float, default=10.0, help='Weight about color')
        return parent_parser

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
        real_dataset = ImageFolder(self.hparams.real_path, transform=real_transforms)
        dataloaders['real'] = DataLoader(real_dataset, collate_fn=collate_fn)

        original_transforms = T.Compose([
                T.RandomResizedCrop(256, scale=(0.3, 1.0), ratio=(1.0, 1.0)),
                T.ToTensor()
            ])
        original_dataset = ImageFolder(self.hparams.original_path, transform=original_transforms)
        dataloaders['original'] = DataLoader(original_dataset, collate_fn=collate_fn)

        gray_transforms = T.Compose([
                T.Grayscale(num_output_channels=3),
                T.ToTensor()
            ])
        gray_dataset = ImageFolder(self.hparams.real_path, transform=gray_transforms)
        dataloaders['gray'] = DataLoader(gray_dataset, collate_fn=collate_fn)

        smooth_transforms = T.Compose([
                T.GaussianBlur(5, sigma = (2.0, 5.0)),
                T.ToTensor()
            ])
        smooth_dataset = ImageFolder(self.hparams.real_path, transform=smooth_transforms)
        dataloaders['smooth'] = DataLoader(smooth_dataset, collate_fn=collate_fn)

        return dataloaders

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), self.hparams.g_lr)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), self.hparams.d_lr)
        return g_opt, d_opt
