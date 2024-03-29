from torch import nn
from transformers import get_polynomial_decay_schedule_with_warmup
from pytorch_lightning.callbacks.base import Callback
from encoders import *
from argparse import Namespace
from pretrain_datasets import *
from torch.utils.data import DataLoader

import torch
import pytorch_lightning as pl


class PretrainModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
            
        self.args, config, self.tokenizer, self.encoder = setting(args, load_model=True)
        self.loss_func = nn.CosineEmbeddingLoss()
        
        self.save_hyperparameters(args)
        
    def train_dataloader(self):
        train_set = PretrainDataset(self.args)
        ppd = PretrainPadCollate(input_pad_id=self.args.pad_id)
        sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=False)
        train_loader = DataLoader(
            train_set, 
            collate_fn=ppd.pad_collate, 
            batch_size=self.args.batch_size, 
            sampler=sampler, 
            num_workers=self.args.num_workers, 
            pin_memory=True
        )
        
        return train_loader
        
    def forward(self, input_ids_0, input_ids_1, padding_masks_0, padding_masks_1):  
        # input_ids_0: (B, L), input_ids_1: (B, L), padding_masks_0: (B, L), padding_masks_1: (B, L)
        
        hidden_states_0 = self.encoder(input_ids=input_ids_0, attention_mask=padding_masks_0)[0]  # (B, L, d_h)
        hidden_states_1 = self.encoder(input_ids=input_ids_1, attention_mask=padding_masks_1)[0]  # (B, L, d_h)
        
        if self.args.pooling == 'cls':
            pooled_hidden_states_0 = hidden_states_0[:, 0, :]  # (B, d_h)
            pooled_hidden_states_1 = hidden_states_1[:, 0, :]  # (B, d_h)
        elif self.args.pooling == 'mean':
            pooled_hidden_states_0 = torch.mean(hidden_states_0, dim=1)  # (B, d_h)
            pooled_hidden_states_1 = torch.mean(hidden_states_1, dim=1)  # (B, d_h)
        elif self.args.pooling == 'max':
            pooled_hidden_states_0 = torch.max(hidden_states_0, dim=1).values  # (B, d_h)
            pooled_hidden_states_1 = torch.max(hidden_states_1, dim=1).values  # (B, d_h)
            
        return pooled_hidden_states_0, pooled_hidden_states_1  # (B, d_h), (B, d_h)
    
    def training_step(self, batch, batch_idx):
        input_ids_0, input_ids_1, labels = batch  # input_ids_0: (B, L), input_ids_1: (B, L) labels: (B)
        padding_masks_0 = (input_ids_0 != self.args.pad_id).float()  # (B, L)
        padding_masks_1 = (input_ids_1 != self.args.pad_id).float()  # (B, L)
        
        outputs_0, outputs_1 = self.forward(input_ids_0, input_ids_1, padding_masks_0, padding_masks_1)  # (B, d_h), (B, d_h)
        
        loss = self.loss_func(outputs_0, outputs_1, labels)
        
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        num_training_samples = len(self.train_dataloader())
        num_devices = len(self.args.gpus) * self.args.num_nodes
        q, r = divmod(num_training_samples, num_devices)
        num_batches = q if r == 0 else q+1
        self.args.num_training_steps = self.args.num_epochs * num_batches
        self.args.num_warmup_steps = int(self.args.warmup_ratio * self.args.num_training_steps)
                
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        scheduler = {
            'scheduler': get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.args.num_training_steps,
                lr_end=1e-6,
                power=2.0,
            ),
            'name': 'learning_rate',
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]

    
class CustomModelCheckpoint(Callback):
    def __init__(self,
                 every_n_train_steps,
                 save_weights_only,
    ):
        super().__init__()
        self.every_n_train_steps = every_n_train_steps
        self.save_weights_only = save_weights_only
        
    def on_batch_end(self, trainer, pl_module):
        step = pl_module.global_step
        if (step+1) % self.every_n_train_steps == 0:
            self.save_step_checkpoint(step, trainer, pl_module)
            
    def on_epoch_end(self, trainer, pl_module):
        step = pl_module.global_step
        self.save_step_checkpoint(step, trainer, pl_module)
    
    def save_step_checkpoint(self, step, trainer, pl_module):
        epoch = pl_module.current_epoch
        pooling = pl_module.args.pooling
        step = pl_module.global_step
        
        metrics = trainer.callback_metrics
        loss = round(metrics['train_loss'].item(), 6)

        ckpt_file = f"dialogue_sentbert_{pooling}_epoch={epoch}_step={step}_train_loss={loss}.ckpt"
        log_dir = trainer.log_dir
        trainer.save_checkpoint(f"{log_dir}/checkpoints/{ckpt_file}", weights_only=self.save_weights_only)    
