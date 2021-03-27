from torch import nn as nn
from torch.nn import functional as F
from pytorch_lightning.core.lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from .encoders import *
from .output_layers import *
from utils import *
from tensorboardX import SummaryWriter

import torch


class BasicLightningModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.save_hyperparameters(args)
        
        fix_seed(self.args.model_seed)
        self.encoder, self.args = load_encoder(self.args)
        
        fix_seed(self.args.model_seed)
        self.output_layer, self.args = load_output_layer(self.args)
        
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1) if self.args.task == 'entity recognition' else nn.BCEWithLogitsLoss()
        
    def forward(self, input_ids, padding_masks=None):  # input_ids: (B, L), padding_masks: (B, L)
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=padding_masks)[0]  # (B, L, d_h)
        
        if self.args.task == 'action prediction':
            hidden_states = hidden_states[:, 0, :]  # (B, d_h)
            
        return self.output_layer(hidden_states)  # (B, L, C) or  (B, C)
    
    def training_step(self, batch, batch_idx):
        input_ids, labels = batch  # input_ids: (B, L), labels: (B, L, C) or (B, C)
        padding_masks = (input_ids != self.args.pad_id).float()  # (B, L)
        
        outputs = self.forward(input_ids, padding_masks)  # (B, L ,C) or (B, C)
        
        if self.args.task == 'entity recognition':
            loss = self.loss_func(outputs.view(-1, self.args.num_classes), labels.view(-1))  # ()
            preds, trues = self.get_entity_results(outputs, labels)
        elif self.args.task == 'action prediction':
            loss = self.loss_func(outputs, labels.float())  # ()
            preds, trues = self.get_action_results(outputs, labels)
            
        return {'loss': loss, 'preds': preds, 'trues': trues}
    
    def training_epoch_end(self, training_step_outputs):
        train_losses = []
        train_preds = []
        train_trues = []
        
        for result in training_step_outputs:
            train_losses.append(result['loss'].item())
            train_preds += result['preds']
            train_trues += result['trues']
        
        if self.args.task == 'entity recognition':
            scores = entity_scores(train_preds, train_trues, self.args.class_dict, round_num=4)
        elif self.args.task == 'action prediction':
            scores = action_scores(train_preds, train_trues, round_num=4)
        
        self.log('train_loss', np.mean(train_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric, value in scores.items():
            self.log(f"train_{metric}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch  # input_ids: (B, L), labels: (B, L, C) or (B, C)
        padding_masks = (input_ids != self.args.pad_id).float()  # (B, L)
        
        outputs = self.forward(input_ids, padding_masks)  # (B, L ,C) or (B, C)
        
        if self.args.task == 'entity recognition':
            loss = self.loss_func(outputs.view(-1, self.args.num_classes), labels.view(-1))  # ()
            preds, trues = self.get_entity_results(outputs, labels)
        elif self.args.task == 'action prediction':
            loss = self.loss_func(outputs, labels.float())  # ()
            preds, trues = self.get_action_results(outputs, labels)
            
        return {'valid_loss': loss, 'preds': preds, 'trues': trues}
    
    def validation_epoch_end(self, validation_step_outputs):
        valid_losses = []
        valid_preds = []
        valid_trues = []
        
        for result in validation_step_outputs:
            valid_losses.append(result['valid_loss'].item())
            valid_preds += result['preds']
            valid_trues += result['trues']
        
        if self.args.task == 'entity recognition':
            scores = entity_scores(valid_preds, valid_trues, self.args.class_dict, round_num=4)
        elif self.args.task == 'action prediction':
            scores = action_scores(valid_preds, valid_trues, round_num=4)
        
        self.log('valid_loss', np.mean(valid_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric, value in scores.items():
            self.log(f"valid_{metric}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    def test_step(self, batch, batch_idx):
        input_ids, labels = batch  # input_ids: (B, L), labels: (B, L, C) or (B, C)
        padding_masks = (input_ids != self.args.pad_id).float()  # (B, L)
        
        outputs = self.forward(input_ids, padding_masks)  # (B, L ,C) or (B, C)
        
        if self.args.task == 'entity recognition':
            loss = self.loss_func(outputs.view(-1, self.args.num_classes), labels.view(-1))  # ()
            preds, trues = self.get_entity_results(outputs, labels)
        elif self.args.task == 'action prediction':
            loss = self.loss_func(outputs, labels.float())  # ()
            preds, trues = self.get_action_results(outputs, labels)
            
        return {'test_loss': loss, 'preds': preds, 'trues': trues}
    
    def test_epoch_end(self, test_step_outputs):
        test_losses = []
        test_preds = []
        test_trues = []
        
        for result in test_step_outputs:
            test_losses.append(result['test_loss'].item())
            test_preds += result['preds']
            test_trues += result['trues']
        
        if self.args.task == 'entity recognition':
            scores = entity_scores(test_preds, test_trues, self.args.class_dict, round_num=4)
        elif self.args.task == 'action prediction':
            scores = action_scores(test_preds, test_trues, round_num=4)
        
        self.log('test_loss', np.mean(test_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric, value in scores.items():
            self.log(f"test_{metric}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def get_entity_results(self, outputs, labels):
        _, preds = torch.max(outputs, dim=-1)  # (B, L)
        
        preds = preds.tolist()
        trues = labels.tolist()
        true_labels = (labels != -1).tolist()
        spots = [(label.index(True), len(label)-list(reversed(label)).index(True)) for label in true_labels]
        preds = [pred[spots[p][0]:spots[p][1]] for p, pred in enumerate(preds)]
        trues = [true[spots[t][0]:spots[t][1]] for t, true in enumerate(trues)]

        return preds, trues
        
    def get_action_results(self, outputs, labels):
        preds = (torch.sigmoid(outputs) > self.args.sigmoid_threshold).long().tolist()
        trues = labels.long().tolist()

        return preds, trues
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
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
