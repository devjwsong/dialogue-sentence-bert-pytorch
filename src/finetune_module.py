from torch import nn as nn
from transformers import get_linear_schedule_with_warmup
from encoders import *
from utils import *
from pytorch_lightning import seed_everything
from argparse import Namespace

import torch
import pytorch_lightning as pl


loss_funcs = {
    'intent': nn.CrossEntropyLoss(),
    'action': nn.BCEWithLogitsLoss()
}


class FinetuneModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = Namespace(**args)
            
        self.args, config, self.tokenizer, self.encoder = setting(args, load_model=True)
        
        seed_everything(self.args.seed, workers=True)
        self.output_layer = nn.Linear(self.args.hidden_size, self.args.num_classes)
        nn.init.xavier_uniform_(self.output_layer.weight)
        
        self.loss_func = loss_funcs[self.args.task]
        
        self.save_hyperparameters(args)
        
    def forward(self, input_ids, padding_masks=None):  # input_ids: (B, L), padding_masks: (B, L)
        hidden_states = self.encoder(input_ids=input_ids, attention_mask=padding_masks)[0]  # (B, L, d_h)
        
        if self.args.pooling == 'cls':
            hidden_states = hidden_states[:, 0, :]  # (B, d_h)
        elif self.args.pooling == 'mean':
            hidden_states = torch.mean(hidden_states, dim=1)  # (B, d_h)
        elif self.args.pooling == 'max':
            hidden_states = torch.max(hidden_states, dim=1).values  # (B, d_h)
            
        return self.output_layer(hidden_states)  # (B, C)
    
    def make_masks(self, input_ids):
        return (input_ids != self.args.pad_id).float()  # (B, L)
    
    def training_step(self, batch, batch_idx):
        input_ids, labels = batch  # input_ids: (B, L), labels: (B) or (B, C)
        padding_masks = self.make_masks(input_ids)  # (B, L)
        
        outputs = self.forward(input_ids, padding_masks)  # (B, C)
        
        if self.args.task == 'intent':
            loss = self.loss_func(outputs, labels)  # ()
        elif self.args.task == 'action':
            loss = self.loss_func(outputs, labels.float())  # ()
        
        return {'loss': loss, 'preds': outputs.detach(), 'trues': labels.detach()}
    
    def training_epoch_end(self, training_step_outputs):
        train_losses = []
        train_preds = []
        train_trues = []
        
        for result in training_step_outputs:
            train_losses.append(result['loss'].item())
            train_preds.append(result['preds'])
            train_trues.append(result['trues'])
        
        if self.args.task == 'intent':
            intent_class_dict = None
            if self.args.dataset == 'oos':
                intent_class_dict = self.args.class_dict
            train_preds, train_trues = self.get_intent_results(train_preds, train_trues)
            scores = intent_scores(train_preds, train_trues, intent_class_dict, round_num=4)
        elif self.args.task == 'action':
            train_preds, train_trues = self.get_action_results(train_preds, train_trues)
            scores = action_scores(train_preds, train_trues, round_num=4)
        
        self.log('train_loss', np.mean(train_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric, value in scores.items():
            self.log(f"train_{metric}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch  # input_ids: (B, L), labels: (B) or (B, C)
        padding_masks = self.make_masks(input_ids)  # (B, L)
        
        outputs = self.forward(input_ids, padding_masks)  # (B, C)
        
        if self.args.task == 'intent':
            loss = self.loss_func(outputs, labels)  # ()
        elif self.args.task == 'action':
            loss = self.loss_func(outputs, labels.float())  # ()
            
        return {'valid_loss': loss, 'preds': outputs.detach(), 'trues': labels.detach()}
    
    def validation_epoch_end(self, validation_step_outputs):
        valid_losses = []
        valid_preds = []
        valid_trues = []
        
        for result in validation_step_outputs:
            valid_losses.append(result['valid_loss'].item())
            valid_preds.append(result['preds'])
            valid_trues.append(result['trues'])
        
        if self.args.task == 'intent':
            intent_class_dict = None
            if self.args.dataset == 'oos':
                intent_class_dict = self.args.class_dict
            valid_preds, valid_trues = self.get_intent_results(valid_preds, valid_trues)
            scores = intent_scores(valid_preds, valid_trues, intent_class_dict, round_num=4)
        elif self.args.task == 'action':
            valid_preds, valid_trues = self.get_action_results(valid_preds, valid_trues)
            scores = action_scores(valid_preds, valid_trues, round_num=4)
        
        self.log('valid_loss', np.mean(valid_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric, value in scores.items():
            self.log(f"valid_{metric}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    def test_step(self, batch, batch_idx):
        input_ids, labels = batch  # input_ids: (B, L), labels: (B) or (B, C)
        padding_masks = self.make_masks(input_ids)  # (B, L)
        
        outputs = self.forward(input_ids, padding_masks)  # (B, C)
        
        if self.args.task == 'intent':
            loss = self.loss_func(outputs, labels)  # ()
        elif self.args.task == 'action':
            loss = self.loss_func(outputs, labels.float())  # ()
            
        return {'test_loss': loss, 'preds': outputs.detach(), 'trues': labels.detach()}
    
    def test_epoch_end(self, test_step_outputs):
        test_losses = []
        test_preds = []
        test_trues = []
        
        for result in test_step_outputs:
            test_losses.append(result['test_loss'].item())
            test_preds.append(result['preds'])
            test_trues.append(result['trues'])
        
        if self.args.task == 'intent':
            intent_class_dict = None
            if self.args.dataset == 'oos':
                intent_class_dict = self.args.class_dict
            test_preds, test_trues = self.get_intent_results(test_preds, test_trues)
            scores = intent_scores(test_preds, test_trues, intent_class_dict, round_num=4)
        elif self.args.task == 'action':
            test_preds, test_trues = self.get_action_results(test_preds, test_trues)
            scores = action_scores(test_preds, test_trues, round_num=4)
        
        self.log('test_loss', np.mean(test_losses), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric, value in scores.items():
            self.log(f"test_{metric}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    def get_intent_results(self, preds, trues):
        new_preds, new_trues = [], []
        for i in range(len(preds)):
            pred, true = preds[i], trues[i]  # (B, C), (B)
            _, pred = torch.max(pred, dim=-1)  # (B)
            new_preds += pred.tolist()
            new_trues += true.tolist()
        
        assert len(new_preds) == len(new_trues)
        
        return new_preds, new_trues
        
    def get_action_results(self, preds, trues):
        new_preds, new_trues = [], []
        for i in range(len(preds)):
            pred, true = preds[i], trues[i]  # (B, C), (B, C)
            new_preds += (torch.sigmoid(pred) > self.args.sigmoid_threshold).long().tolist()
            new_trues += true.long().tolist()
        
        assert len(new_preds) == len(new_trues)
        
        return new_preds, new_trues
    
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
