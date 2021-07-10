from torch import nn as nn
from transformers import get_linear_schedule_with_warmup
from .encoders import *
from utils import *
from pytorch_lightning import seed_everything
from argparse import Namespace

import torch
import pytorch_lightning as pl


class PretrainModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
            
        self.args, config, self.tokenizer, self.encoder = setting(args, load_model=True)
        
        seed_everything(self.args.seed, workers=True)
        self.class_layer = nn.Linear(3 * self.args.hidden_size, self.args.num_classes)
        self.lm_layer = nn.Linear(self.args.hidden_size, self.args.vocab_size)
        
        self.class_loss_func = nn.CrossEntropyLoss()
        self.lm_loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.save_hyperparameters(args)
        
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
            
        dists = torch.abs(pooled_hidden_states_0 - pooled_hidden_states_1)  # (B, d_h)
        total_hidden_states = torch.cat((pooled_hidden_states_0, pooled_hidden_states_1, dists), dim=-1)  # (B, 3d_h)
            
        return self.class_layer(total_hidden_states), self.lm_layer(hidden_states_1)  # (B, C), (B, L, V)
    
    def training_step(self, batch, batch_idx):
        # input_ids_0: (B, L), input_ids_1: (B, L) class_labels: (B), lm_labels: (B, L)
        input_ids_0, input_ids_1, class_labels, lm_labels = batch  
        padding_masks_0 = (input_ids_0 != self.args.pad_id).float()  # (B, L)
        padding_masks_1 = (input_ids_1 != self.args.pad_id).float()  # (B, L)
        
        class_outputs, lm_outputs = self.forward(input_ids_0, input_ids_1, padding_masks_0, padding_masks_1)  # (B, C), (B, L, V)
        
        class_loss = self.class_loss_func(class_outputs, class_labels)
        preds, trues = self.get_results(class_outputs, class_labels)
        
        lm_loss = self.lm_loss_func(lm_outputs.view(-1, self.args.vocab_size), lm_labels.view(-1))
        loss = class_loss + self.args.mlm_factor * lm_loss
            
        return {'loss': loss, 'preds': preds, 'trues': trues}
    
    def training_epoch_end(self, training_step_outputs):
        train_losses = []
        train_preds = []
        train_trues = []
        
        for result in training_step_outputs:
            train_losses.append(result['loss'].item())
            train_preds += result['preds']
            train_trues += result['trues']
            
        scores = pretrain_scores(train_preds, train_trues, round_num=4)
        
        self.log('train_loss', np.mean(train_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric, value in scores.items():
            self.log(f"train_{metric}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    def validation_step(self, batch, batch_idx):
        # input_ids_0: (B, L), input_ids_1: (B, L) class_labels: (B), lm_labels: (B, L)
        input_ids_0, input_ids_1, class_labels, lm_labels = batch  
        padding_masks_0 = (input_ids_0 != self.args.pad_id).float()  # (B, L)
        padding_masks_1 = (input_ids_1 != self.args.pad_id).float()  # (B, L)
        
        class_outputs, lm_outputs = self.forward(input_ids_0, input_ids_1, padding_masks_0, padding_masks_1)  # (B, C), (B, L, V)
        
        class_loss = self.class_loss_func(class_outputs, class_labels)
        preds, trues = self.get_results(class_outputs, class_labels)
        
        lm_loss = self.lm_loss_func(lm_outputs.view(-1, self.args.vocab_size), lm_labels.view(-1))
        loss = class_loss + self.args.mlm_factor * lm_loss
            
        return {'valid_loss': loss, 'preds': preds, 'trues': trues}
    
    def validation_epoch_end(self, validation_step_outputs):
        valid_losses = []
        valid_preds = []
        valid_trues = []
        
        for result in validation_step_outputs:
            valid_losses.append(result['valid_loss'].item())
            valid_preds += result['preds']
            valid_trues += result['trues']
        
        scores = pretrain_scores(valid_preds, valid_trues, round_num=4)
        
        self.log('valid_loss', np.mean(valid_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric, value in scores.items():
            self.log(f"valid_{metric}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def get_results(self, outputs, labels):
        _, preds = torch.max(outputs, dim=-1)  # (B)
        
        preds = preds.tolist()
        trues = labels.tolist()
        
        assert len(preds) == len(trues)
        
        return preds, trues
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        if self.args.warmup_steps < 0.0:
            return [optimizer]
        else:
            scheduler = {
                'scheduler': get_linear_schedule_with_warmup(
                                optimizer, 
                                num_warmup_steps=self.args.warmup_steps, 
                                num_training_steps=self.args.total_train_steps
                            ),
                'name': 'learning_rate',
                'interval': 'step',
                'frequency': 1

            }

            return [optimizer], [scheduler]
