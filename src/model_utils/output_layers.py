from torch import nn as nn

import torch


def load_output_layer(args):
    if args.task == 'entity recognition':
        er_layer = EntityRecognition(args)
        er_layer.init_params()

        return er_layer, args

    elif args.task == 'action prediction':        
        ap_layer = ActionPrediction(args)
        ap_layer.init_params()

        return ap_layer, args
        
        
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
        