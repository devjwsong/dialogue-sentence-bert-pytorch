from torch import nn as nn
from .transformer_modules import *
from .output_layers import *

import torch
import numpy as np
import random


def load_model(args, class_dict=None):
    assert 'bert' in args.model_name.lower() or 'gpt' in args.model_name.lower(), "Invalid model name."
    
    if class_dict is None:
        args.model_name = 'bert-base-uncased'
        tokenizer, encoder, args = load_encoder(args)
        
        encoder_embedding = encoder.embeddings
        if args.decoder == 'gru' :
            lm_decoder = GruDecoder(args, encoder_embedding)
        elif args.decoder == 'transformer':
            lm_decoder = TransformerDecoder(args, encdoer_embedding)
            
        output_layer = LanguageModeling(args)
        model = PretrainModel()
        
    else:
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


class PretrainModel(nn.Module):
    def __init__(self, args, encoder, decoder, output_layer):
        super(PretrainModel, self).__init__() 
        
        # Random seed fixing
        np.random.seed(args.model_seed)
        torch.manual_seed(args.model_seed)
        torch.cuda.manual_seed_all(args.model_seed)
        random.seed(args.model_seed)
        
        self.decoder_type = args.decoder
        
        self.encoder = encoder
        self.decoder = decoder
        self.lm_layer = output_layer
        
    def forward(self, src_input_ids, encoder_masks, trg_input_ids, trg_lens=None, nopeak_masks=None, trg_padding_masks=None):
        encoder_outputs = self.encoder(src_input_ids, attention_mask=encoder_masks)[0]  # (B, L, d_h)
        
        if trg_lens is not None:
            decoder_outputs = self.decoder(
                trg_input_ids.transpose(0, 1), 
                encoder_outputs.transpose(0, 1),
                encoder_masks,
                trg_lens=trg_lens.transpose(0, 1),
            )  # (L, B, d_h)
        elif nopeak_masks is not None and trg_padding_masks is not None:
            decoder_outputs = self.decoder(
                trg_input_ids.transpose(0, 1), 
                encoder_outputs.transpose(0, 1),
                encoder_masks,
                nopeak_masks=nopeak_masks,
                trg_padding_masks=trg_padding_masks
            )  # (L, B, d_h)
            
        outputs = self.lm_layer(decoder_outputs.transpose(0, 1))  # (B, T_L, V)
        
        return outputs


class FineTuneModel(nn.Module):
    def __init__(self, pretrained_model, output_layer, pooling=None):
        super(FineTuneModel, self).__init__()
        
        # Random seed fixing
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        random.seed(0)
        
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
        