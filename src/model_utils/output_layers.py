from torch import nn as nn
from torch.nn import functional as F
from tqdm import tqdm

import torch


def load_finetune_layer(args, class_dict=None, pretrained_model=None, tokenizer=None):
    if args.task == 'intent detection':
        args.num_classes = len(class_dict)

        id_layer = IntentDetection(args)
        id_layer.init_params()

        return id_layer, args

    elif args.task == 'entity recognition':
        args.num_classes = len(class_dict)

        er_layer = EntityRecognition(args)
        er_layer.init_params()

        return er_layer, args

    elif args.task == 'action prediction':
        args.num_classes = len(class_dict)

        ap_layer = ActionPrediction(args)
        ap_layer.init_params()

        return ap_layer, args
        

class LanguageModeling(nn.Module):
    def __init__(self, args):
        super(LanguageModeling, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.vocab_size = args.vocab_size
        
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(self, hiddens):
        # trg_hiddens: (B, L, d_h)
        
        return self.linear(trg_hiddens)  # (B, L, V)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        
        
class IntentDetection(nn.Module):
    def __init__(self, args):
        super(IntentDetection, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        
    def forward(self, trg_hiddens):
        # trg_hiddens: (B, d_h) ([CLS] or mean)
        
        return self.linear(trg_hiddens)  # (B, C)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        
        
class EntityRecognition(nn.Module):
    def __init__(self, args):
        super(EntityRecognition, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        
    def forward(self, hiddens):
        # hiddens: (B, L, d_h)
        
        return self.linear(hiddens)  # (B, L, C)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        
        
class ActionPrediction(nn.Module):
    def __init__(self, args):
        super(ActionPrediction, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.num_classes = args.num_classes
        
        self.linear = nn.Linear(self.hidden_size, self.num_classes)
        
    def forward(self, trg_hiddens):
        # trg_hiddens: (B, d_h) ([CLS] or mean)
        
        return self.linear(trg_hiddens)  # (B, C)
    
    def init_params(self):
        nn.init.xavier_uniform_(self.linear.weight)
        