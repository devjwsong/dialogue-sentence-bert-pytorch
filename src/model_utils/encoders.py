from transformers import *
from torch import nn


def load_encoder(args):
    bert = BertModel.from_pretrained(args.model_name)
    bert.resize_token_embeddings(args.vocab_size)
    
    if 'bert' in args.model_name.lower():
        encoder = bert
    else:
        # TODO GRU
        encoder = None
    
    return encoder, args


# TODO
class ContextEncoder(nn.Module):
    def __init__(self, args):
        pass
    
    def forward(self):
        pass
    