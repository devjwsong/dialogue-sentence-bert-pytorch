from transformers import *
from torch import nn

import json
import os
import torch


# [model_id, config, tokenizer, model, cls_token, sep_token, pad_token, unk_token]
attr_map = {
    'bert': ['bert-base-uncased', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'albert': ['albert-base-v1', AlbertConfig, AlbertTokenizer, AlbertModel, '[CLS]', '[SEP]', '<pad>', '<unk>'],
    'distilbert': ['distilbert-base-uncased', DistilBertConfig, DistilBertTokenizer, DistilBertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'convbert': ['convbert', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'todbert': ['TODBERT/TOD-BERT-JNT-V1', AutoConfig, AutoTokenizer, AutoModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'bert-teacher': ['bert-base-uncased', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'albert-teacher': ['albert-base-v1', AlbertConfig, AlbertTokenizer, AlbertModel, '[CLS]', '[SEP]', '<pad>', '<unk>'],
    'distilbert-teacher': ['distilbert-base-uncased', DistilBertConfig, DistilBertTokenizer, DistilBertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'convbert-teacher': ['convbert', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'todbert-teacher': ['TODBERT/TOD-BERT-JNT-V1', AutoConfig, AutoTokenizer, AutoModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'bert-student': ['bert-base-uncased', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'albert-student': ['albert-base-v1', AlbertConfig, AlbertTokenizer, AlbertModel, '[CLS]', '[SEP]', '<pad>', '<unk>'],
    'distilbert-student': ['distilbert-base-uncased', DistilBertConfig, DistilBertTokenizer, DistilBertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'convbert-student': ['convbert', BertConfig, BertTokenizer, BertModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]'],
    'todbert-student': ['TODBERT/TOD-BERT-JNT-V1', AutoConfig, AutoTokenizer, AutoModel, '[CLS]', '[SEP]', '[PAD]', '[UNK]']
}


def setting(args):
    # Set specific encoder model's attributes
    if 'student' in args.model_name:
        config, tokenizer, model, attrs = load_student(args)
    else:
        config, tokenizer, model, attrs = load_teacher(args)
        
    args.cls_token = attrs[4]
    args.sep_token = attrs[5]
    args.pad_token = attrs[6]
    args.unk_token = attrs[7]
    
    vocab = tokenizer.get_vocab()
    args.vocab_size = len(vocab)
    args.cls_id = vocab[args.cls_token]
    args.sep_id = vocab[args.sep_token]
    args.pad_id = vocab[args.pad_token]
    args.unk_id = vocab[args.unk_token]
    
    return args, config, tokenizer, model
    

def load_teacher(args):
    attrs = attr_map[args.model_name]
    
    model_path = attrs[0]
    if 'conv' in args.model_name:
        model_path = f"{args.ckpt_dir}/{args.model_name}"
    
    config = attrs[1].from_pretrained(model_path)
    tokenizer = attrs[2].from_pretrained(model_path)
    model = attrs[3].from_pretrained(model_path)
    args.hidden_size = model.config.hidden_size

    if 'teacher' in args.model_name:
        model.load_state_dict(torch.load(f"{args.ckpt_dir}/{args.model_name}/{args.ckpt_name}"))
        
    return config, tokenizer, model, attrs


def load_student(args):
    attrs = attr_map[args.model_name]
    
    model_path = f"{args.ckpt_dir}/{args.model_name}"
    
    with open(f"{model_path}/student_config.json", 'r') as f:
        config = json.load(f)
    tokenizer = attrs[2].from_pretrained(attrs[0])
    teacher_model = attrs[3].from_pretrained(attrs[0])
    args.embedding_size = teacher_model.config.hidden_size
    args.hidden_size = config['hidden_size']
    args.num_heads = config['num_heads']
    args.num_layers = config['num_layers']
    
    embeddings = teacher_model.embeddings
    model = StudentModel(args, embeddings)
    model.load_state_dict(torch.load(f"{model_path}/{args.ckpt_name}"))
    
    return config, tokenizer, model, attrs

    
class StudentModel(nn.Module):
    def __init__(self, args, embeddings):
        super().__init__()
        
        self.embeddings = embeddings
        self.projection = nn.Linear(args.embedding_size, args.hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=args.num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
        
    def forward(self, input_ids, attention_mask=None):  # input_ids: (B, S_L), padding_masks: (B, S_L)
        input_embs = self.embeddings(input_ids)  # (B, S_L, D_T)
        input_embs = self.projection(input_embs)  # (B, S_L, D_S)
        
        outputs = self.encoder(src=input_embs.transpose(0, 1), src_key_padding_mask=attention_mask)  # (S_L, B, D_S)
        
        return outputs.transpose(0, 1)  # (B, S_L, D_S)
    
