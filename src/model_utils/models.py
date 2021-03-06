from torch import nn as nn
from .transformer_modules import *
from .output_layers import *

import torch


def load_model(args, class_dict=None):
    assert 'bert' in args.model_name.lower() or 'gpt' in args.model_name.lower(), "Invalid model name."
    
    if 'bert' in args.model_name.lower():
        pooling = None if args.task == 'entity recognition' else 'cls'
        tokenizer, pretrained_model, args = load_encoder(args)
    elif 'gpt' in args.model_name.lower():
        pooling = None if args.task == 'entity recognition' else 'mean'
        tokenizer, pretrained_model, args = load_decoder(args)
        
    output_layer, args = load_finetune_layer(args, class_dict=class_dict, pretrained_model=pretrained_model, tokenizer=tokenizer)
    model = FineTuneModel(pretrained_model=pretrained_model, output_layer=output_layer, pooling=pooling)
            
    assert tokenizer is not None
    assert model is not None
    
    return tokenizer, model, args


class FineTuneModel(nn.Module):
    def __init__(self, pretrained_model, output_layer, pooling=None):
        super(FineTuneModel, self).__init__()
        
        self.pretrained_model = pretrained_model
        self.output_layer = output_layer
        self.pooling = pooling
            
    def forward(self, input_ids, padding_masks=None):
        # input_ids: (B, L), padding_masks: (B, L)
        
        hidden_states = self.pretrained_model(input_ids, attention_mask=padding_masks)[0]  # (B, L, d_h)
        if self.pooling == 'cls':
            hidden_states = hidden_states[:, 0, :]  # (B, d_h)
        elif self.pooling == 'mean':
            hidden_states = torch.mean(hidden_states, dim=1)  # (B, d_h)
        
        # Intent Detection: (B, C), Entity Recognition: (B, L, C), Action Prediction: (B, C)
        outputs = self.output_layer(hidden_states)
        
        return outputs
        