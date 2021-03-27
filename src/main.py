from transformers import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import nn
from model_utils.basic_module import *
from data_utils.basic_datasets import *
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import *

import torch
import os, sys
import numpy as np
import argparse
import json
import pickle
import random


def run(args):
    # For directory setting
    args.dataset_dir = f"{args.data_dir}/{args.processed_dir}/{args.dataset}"
    assert os.path.isdir(args.dataset_dir)
    
    whole_setting = f"setting{args.setting}_turn{args.max_turns}"
    task_desc = "entity" if args.task == 'entity recognition' else "action"
    args.ckpt_dir = f"{args.ckpt_dir}/{task_desc}/{args.dataset}/{whole_setting}"
    with open(f"{args.dataset_dir}/{task_desc}_{args.class_dict_name}.json", 'r') as f:
        args.class_dict = json.load(f)
        
    args.num_classes = len(args.class_dict)
        
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        
    # Tokenizer setting
    if args.setting == 1:
        args.model_name = "bert-base-uncased"
        
    bert_config = BertConfig.from_pretrained(args.model_name)
    
    args.max_encoder_len = bert_config.max_position_embeddings
    args.hidden_size = bert_config.hidden_size
        
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    args.bos_token = '[CLS]'
    args.eos_token = '[SEP]'
    args.pad_token = '[PAD]'
    args.speaker1_token = '[USR]'
    args.speaker2_token = '[SYS]'
    
    special_tokens = {
        'additional_special_tokens': [args.speaker1_token, args.speaker2_token]
    }
    
    num_new_tokens = tokenizer.add_special_tokens(special_tokens)
    vocab = tokenizer.get_vocab()
    args.vocab_size = len(vocab)
    
    args.bos_id = vocab[args.bos_token]
    args.eos_id = vocab[args.eos_token]
    args.pad_id = vocab[args.pad_token]
    args.speaker1_id = vocab[args.speaker1_token]
    args.speaker2_id = vocab[args.speaker2_token]
    
    print("Loading datasets...")
    # For data loading
    cached = True if args.cached=='yes' else False
    if args.task == 'entity recognition':
        if args.setting == 0:
            train_set = BasicERDataset(args, args.train_prefix, args.class_dict, tokenizer, cached=cached)
            valid_set = BasicERDataset(args, args.valid_prefix, args.class_dict, tokenizer, cached=cached)
            test_set = BasicERDataset(args, args.test_prefix, args.class_dict, tokenizer, cached=cached)
        else:
            train_set = EffERDataset(args, args.train_prefix, args.class_dict, tokenizer, cached=cached)
            valid_set = EffERDataset(args, args.valid_prefix, args.class_dict, tokenizer, cached=cached)
            test_set = EffERDataset(args, args.test_prefix, args.class_dict, tokenizer, cached=cached)
    elif args.task == 'action prediction':
        if args.setting == 0:
            train_set = BasicAPDataset(args, args.train_prefix, args.class_dict, tokenizer, cached=cached)
            valid_set = BasicAPDataset(args, args.valid_prefix, args.class_dict, tokenizer, cached=cached)
            test_set = BasicAPDataset(args, args.test_prefix, args.class_dict, tokenizer, cached=cached)
        else:
            train_set = EffAPDataset(args, args.train_prefix, args.class_dict, tokenizer, cached=cached)
            valid_set = EffAPDataset(args, args.valid_prefix, args.class_dict, tokenizer, cached=cached)
            test_set = EffAPDataset(args, args.test_prefix, args.class_dict, tokenizer, cached=cached)
    
    args.total_train_steps = int(len(train_set) / args.batch_size * args.num_epochs)
    args.warmup_steps = int(args.total_train_steps * args.warmup_prop)
    
    # Random seed fixing for model
    fix_seed(args.model_seed)
    
    # Model setting
    print(f"Loading model & tokenizer for {task_desc}...")       
    pl_model = BasicLightningModule(args)
    
    # Re-seed random seed for data shuffle
    fix_seed(args.data_seed)
    
    # Dataloaders
    input_pad_id = args.pad_id
    if args.task == 'entity recognition':
        label_pad_id = -1
        ppd = EntityPadCollate(input_pad_id=input_pad_id, label_pad_id=label_pad_id)
    elif args.task == 'action prediction':
        ppd = ActionPadCollate(input_pad_id=input_pad_id)
    
    train_loader = DataLoader(train_set, collate_fn=ppd.pad_collate, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, collate_fn=ppd.pad_collate, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, collate_fn=ppd.pad_collate, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    print("Setting pytorch lightning callback & trainer...")
    # Model checkpoint callback
    if args.task == 'entity recognition':
        filename = "best_ckpt_{epoch}_{train_entity_micro_f1:.4f}_{valid_entity_micro_f1:.4f}"
        monitor = "valid_entity_micro_f1"
    else:
        filename = "best_ckpt_{epoch}_{train_micro_f1:.4f}_{valid_micro_f1:.4f}"
        monitor = "valid_micro_f1"
    
    checkpoint_callback = ModelCheckpoint(
        dirpath =f"{args.ckpt_dir}",
        filename=filename,
        verbose=True,
        monitor=monitor,
        mode='max'
    )
    
    # Trainer setting
    trainer = Trainer(
        checkpoint_callback=checkpoint_callback,
        check_val_every_n_epoch=1,
        gpus=args.gpu,
        auto_select_gpus=True,
        num_nodes=args.num_nodes,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.max_grad_norm
    )
    
    print("Train starts.")
    trainer.fit(model=pl_model, train_dataloader=train_loader, val_dataloaders=valid_loader)
    print("Training done.")
    
    print("Test starts.")
    trainer.test(model=pl_model, test_dataloaders=test_loader, ckpt_path='best')
    
    print("GOOD BYE.")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', required=True, type=str, help="The name of the task.")
    parser.add_argument('--dataset', required=True, type=str, help="The name of the dataset.")
    parser.add_argument('--model_name', required=True, type=str, default="bert-base-uncased", help="The model name to use if you train in setting 0.")
    parser.add_argument('--ckpt_dir', required=True, type=str, default="saved_models", help="The directory path for saved checkpoints.")
    parser.add_argument('--data_dir', required=True, type=str, default="data", help="The parent directory path for data files.")
    parser.add_argument('--processed_dir', required=True, type=str, default="processed", help="The directory path to finetuning data files.")
    parser.add_argument('--class_dict_name', required=True, type=str, default="class_dict", help="The name of class dictionary json file.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of file name related to train set.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of file name related to valid set.")
    parser.add_argument('--test_prefix', type=str, default="test", help="The prefix of file name related to test set.")
    parser.add_argument('--max_turns', required=True, type=int, default=1, help="The maximum number of dialogue turns.")
    parser.add_argument('--num_epochs', type=int, default=1, help="The number of total epochs.")
    parser.add_argument('--batch_size', required=True, type=int, default=1, help="The batch size in one process.")
    parser.add_argument('--num_workers', required=True, type=int, default=1, help="The number of workers for data loading.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="The starting learning rate.")
    parser.add_argument('--warmup_prop', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The max gradient for gradient clipping.")
    parser.add_argument('--sigmoid_threshold', required=True, type=float, default=0.0, help="The sigmoid threshold for action prediction task.")
    parser.add_argument('--cached', required=True, type=str, default='no', help="Using the cached data or not?")
    parser.add_argument('--model_seed', required=True, type=int, default=0, help="The seed number for model initialization.")
    parser.add_argument('--data_seed', required=True, type=int, default=0, help="The seed number for data shuffle.")
    parser.add_argument('--setting', required=True, type=int, default=0, help="Multi-turn setting.")
    
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--num_nodes', type=int, default=1)
    
    args = parser.parse_args()
    
    assert args.setting >= 0 and args.setting < 2, "The setting value must be among 0 ~ 1."
    assert args.task == 'entity recognition'\
        or args.task == 'action prediction', \
        "You must specify a correct finetune task."
    assert args.dataset is not None, "You must specify the dataset name."
    assert 'bert' in args.model_name.lower()
    
    print("#"*50 + "Running spec" + "#"*50)
    print(args)
    
    input("Press Enter to continue...")
    
    run(args)
