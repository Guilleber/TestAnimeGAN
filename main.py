import pytorch_lightning as pl

import torch

import argparse
from datetime import datetime
import sys

from model import AnimeGAN


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--load_weights_from', type=str, default=None)
    parser.add_argument('--save_best_model', action='store_true')
    
    parser.add_argument('--bs', type=int, help="mini-batch size", default=16)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=101)

    parser = AnimeGAN.add_model_specific_args(parser)

    args = parser.parse_args()

    # print to error stream as the logs for the standard stream are often full of junk :)
    print("parameters = {}".format(args), file=sys.stderr)
    print("start time = {}".format(datetime.now().strftime("%d/%m/%Y %H:%M")), file=sys.stderr)

    if args.load_weights_from is None:
        model = AnimeGAN(args)
    else:
        model = AnimeGAN.load_from_checkpoint(args.load_weights_from)

    # reproducibility
    pl.seed_everything(42)

    # saves best model
    callbacks = []
    if args.save_best_model:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='epoch',
                                             dirpath='./saved_models/',
                                             filename=args.exp_name + '-{epoch:02d}-{val_acc:2.2f}',
                                             save_top_k=1,
                                             verbose=True,
                                             mode='max')
        callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(gpus=args.gpus,
                         accelerator='ddp',
                         checkpoint_callback=args.save_best_model,
                         callbacks=callbacks,
                         gradient_clip_val=2.,
                         max_epochs=args.epochs,
                         limit_val_batches=0)

    trainer.fit(model)

    print("end time = {}".format(datetime.now().strftime("%d/%m/%Y %H:%M")), file=sys.stderr)