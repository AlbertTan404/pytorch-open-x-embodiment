import copy
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup

import lightning.pytorch as pl

from train import instantiate_from_config


class ExampleModel(pl.LightningModule):
    def __init__(
        self,
        model_kwargs,
        training_kwargs,
        all_config=None
    ):
        super().__init__()

        self.all_config = all_config
        self.training_kwargs = training_kwargs
        self.model_kwargs = model_kwargs
        self.save_hyperparameters()

        self.action_dim = action_dim = model_kwargs['action_dim']
        self.hidden_size = hidden_size = model_kwargs['hidden_size']
        self.horizon = horizon = model_kwargs['horizon']

        self.action_emb = nn.Linear(action_dim, hidden_size)
        if model_kwargs.get('low_dim_feature_dim') is not None:
            self.low_dim_emb = nn.Linear(model_kwargs['low_dim_feature_dim'], hidden_size)
        else:
            self.low_dim_emb = None
        self.language_emb = nn.Linear(in_features=model_kwargs['language_feature_dim'], out_features=hidden_size)
        self.img_emb = nn.Linear(in_features=512, out_features=hidden_size)

        self.action_head = nn.Linear(hidden_size, action_dim)

    def configure_optimizers(self):
        kwargs = self.training_kwargs
        tuned_parameters = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            tuned_parameters,
            lr=kwargs.lr,
        )
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=kwargs.warmup_steps, num_training_steps=kwargs.num_training_steps)

        self.lr_scheduler = scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
    
    def get_img_emb(self, raw_image_features):
        return self.img_emb(raw_image_features)

    def get_language_emb(self, raw_language_features):
        return self.language_emb(raw_language_features)
    
    def get_low_dim_emb(self, raw_low_dim_data=None):
        if raw_low_dim_data is None or self.low_dim_emb is None:
            return None
        return self.low_dim_emb(raw_low_dim_data)

    def forward(self, batch, batch_idx, sample_posterior=True, split='train'):
        action = batch['action']
        language_emb = self.get_language_emb(batch['language'])
        img_emb = self.get_img_emb(batch['image'])
        low_dim_emb = self.get_low_dim_emb(batch.get('low_dim'))

        batch_size = action.shape[0]
        pred_action = action

        loss = F.mse_loss(action, pred_action)
        loss_log = {
            f'{split}/loss': loss
        }
        return loss, loss_log
    
    def training_step(self, batch, batch_idx):
        self.last_training_batch = batch
        total_loss, log_dict = self.forward(batch=batch, batch_idx=batch_idx)
        self.log_dict(log_dict, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, log_dict = self.forward(batch=batch, batch_idx=batch_idx, split='val')
        self.log_dict(log_dict, sync_dist=True)
        return total_loss
