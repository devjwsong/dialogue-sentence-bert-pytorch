from transformers import *
from torch import nn

import json
import os
import torch


# [model_id, config, tokenizer, model, cls_token, sep_token, pad_token, unk_token]
attr_map = {
    'bert': ['bert-base-uncased', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'convbert': ['convbert', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'todbert': ['TODBERT/TOD-BERT-JNT-V1', AutoConfig, AutoTokenizer, AutoModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'sentbert-cls': ['sentence-transformers/bert-base-nli-cls-token', AutoConfig, AutoTokenizer, AutoModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'sentbert-mean': ['sentence-transformers/bert-base-nli-mean-tokens', AutoConfig, AutoTokenizer, AutoModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'sentbert-max': ['sentence-transformers/bert-base-nli-max-tokens', AutoConfig, AutoTokenizer, AutoModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'dialogsentbert-cls': ['bert-base-uncased', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'dialogsentbert-mean': ['bert-base-uncased', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'dialogsentbert-max': ['bert-base-uncased', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'dialogsentconvbert-cls': ['convbert', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'dialogsentconvbert-mean': ['convbert', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'dialogsentconvbert-max': ['convbert', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'dialogsenttodbert-cls': ['TODBERT/TOD-BERT-JNT-V1', AutoConfig, AutoTokenizer, AutoModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'dialogsenttodbert-mean': ['TODBERT/TOD-BERT-JNT-V1', AutoConfig, AutoTokenizer, AutoModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
    'dialogsenttodbert-max': ['TODBERT/TOD-BERT-JNT-V1', AutoConfig, AutoTokenizer, AutoModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]', '[MASK]'],
}


def setting(args, load_model=True):
    # Set specific encoder model's attributes
    config, tokenizer, model, attrs = load_encoder(args, load_model)
        
    args.cls_token = attrs[4]
    args.sep_token = attrs[5]
    args.pad_token = attrs[6]
    args.unk_token = attrs[7]
    args.mask_token = attrs[8]
    
    vocab = tokenizer.get_vocab()
    args.vocab_size = len(vocab)
    args.cls_id = vocab[args.cls_token]
    args.sep_id = vocab[args.sep_token]
    args.pad_id = vocab[args.pad_token]
    args.unk_id = vocab[args.unk_token]
    args.mask_id = vocab[args.mask_token]
    
    return args, config, tokenizer, model
    

def load_encoder(args, load_model):
    attrs = attr_map[args.model_name]
    
    model_dir = attrs[0]
    if 'conv' in model_dir:
        model_dir = f"{args.ckpt_dir}/{model_dir.split('-')[0]}"
    
    config = attrs[1].from_pretrained(model_dir)
    tokenizer = attrs[2].from_pretrained(model_dir)
    model = None
    if load_model:
        model = attrs[3].from_pretrained(model_dir)
        args.hidden_size = model.config.hidden_size

        if 'dialogsent' in args.model_name:
            model.load_state_dict(torch.load(f"{args.ckpt_dir}/{args.model_name}/{args.ckpt_name}.pt"))
        
    return config, tokenizer, model, attrs
