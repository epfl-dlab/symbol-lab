from typing import Any, Dict, List, Callable, Union, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')
import transformers

import torch
from torch import optim
import pytorch_lightning as pl
import os
import wandb

from .vqvae_model import VQVAE

class VQVAEPl(pl.LightningModule):

    def __init__(self,
                 *args,
                 **kwargs) -> None:
        super().__init__()
        
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = VQVAE(**self.hparams)
        self.tokenizer = None
        
    def forward(self, *args, **kwargs) -> Tensor:
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        
        '''
        batch: [batch_size, seq_len+2 (bos, eos)]
            each sample is a sequence of [bos, token, token, eos]
            i.e. [0, 2, 3, 1]
        '''
        
        batch = torch.transpose(batch, 0, 1)

        # Removing the BOS tokens from target sequences. # target: [seq_len+1, batch_size] (seq_len+EOS)
        # target = [seq_len+1, batch_size]
        target = batch[1:, :]

        # Removing the EOS tokens from source sequences. # source: [seq_len+1, batch_size] (BOS+seq_len)
        source = batch[:-1, :]
        
        results = self.forward(source, target, teacher_forcing_ratio=0.5)
        train_loss = self.model.loss_function(*results)
        
        # train_loss contains 'loss', 'Reconstruction_Loss', 'VQ_Loss'
        wandb.log({key+'_train': val.item() for key, val in train_loss.items()})
        
        # There's no need to return a dictionary with the key 'loss'. train_loss already 
        # has such key
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        
        batch = torch.transpose(batch, 0, 1)

        # Removing the BOS tokens from target sequences. # target: [seq_len+1, batch_size] (seq_len+EOS)
        # target = [seq_len+1, batch_size]
        target = batch[1:, :]

        # Removing the EOS tokens from source sequences. # source: [seq_len+1, batch_size] (BOS+seq_len)
        source = batch[:-1, :]

        results = self.forward(source, target, teacher_forcing_ratio=0.0)
        val_loss = self.model.loss_function(*results)
        
        # Printing two example source sequences and their reconstructions
        print(f"\nsource grid   :{batch[1:,3]}")
        print(f"predicted grid:{results[0].argmax(2, keepdim=False).squeeze()[:,3]}")
        
        print(f"\nsource grid   :{batch[1:,2]}")
        print(f"predicted grid:{results[0].argmax(2, keepdim=False).squeeze()[:,2]}")
        
        # train_loss contains 'loss', 'Reconstruction_Loss', 'VQ_Loss'
        wandb.log({key+'_valid': val.item() for key, val in val_loss.items()})
        
        # Required for early stopping callback
        self.log("val_loss", val_loss['loss'].item())

        return val_loss

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        
        return


    def configure_optimizers(self):

        # Apply weight decay to all parameters except for the biases and the weight for Layer Normalization
        no_decay = ["bias"]

        # Per-parameter optimization.
        # Each dict defines a parameter group and contains the list of parameters to be optimized in a key `params`
        # Other keys should match keyword arguments accepted by the optimizers and
        # will be used as optimization params for the parameter group
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
#                 "betas": self.hparams.adam_betas,
                "betas": (0.9, 0.999),
                "eps": self.hparams.adam_eps,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
#                 "betas": self.hparams.adam_betas,
                "betas": (0.9, 0.999),
                "eps": self.hparams.adam_eps,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        
#         optimizer = torch.optim.Adam(
#             optimizer_grouped_parameters,
#             lr=self.hparams.lr,
# #             weight_decay=self.hparams.weight_decay,
#         )

        if self.hparams.schedule_name == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_updates,
                # num_training_steps=self.hparams.total_num_updates,
            )
        elif self.hparams.schedule_name == "polynomial":
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_updates,
                num_training_steps=self.hparams.total_num_updates,
                lr_end=self.hparams.lr_end,
            )
        elif self.hparams.schedule_name == "reduce_lr_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=1,
                threshold=0.01,
            )

        lr_dict = {
            "scheduler": scheduler,  # scheduler instance
            "interval": "epoch",  # The unit of the scheduler's step size. 'step' or 'epoch
            "frequency": 1,  # corresponds to updating the learning rate after every `frequency` epoch/step
            "name": f"LearningRateScheduler-{self.hparams.schedule_name}"
            # Used by a LearningRateMonitor callback
            ,"monitor": "val_loss"
        }

        return [optimizer], [lr_dict]
    

