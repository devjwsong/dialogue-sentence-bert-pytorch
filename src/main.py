from transformers import *
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from torch.nn import functional as F
from model_utils.models import *
from data_utils.custom_datasets import *
from metrics import *

import torch
import torch.distributed as dist
import os, sys
import numpy as np
import argparse
import json
import pickle
import torch.multiprocessing as mp


def main(gpu, args):
    args.rank = args.rank * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )
    torch.cuda.set_device(gpu)
    args.device = torch.device(f"cuda:{gpu}")
    
    # For data setting
    args.processed_dir = f"{args.data_dir}/{args.processed_dir}"
    if args.task == 'entity recognition':
        args.dataset_dir = f"{args.processed_dir}/entity/{args.dataset}"
        args.ckpt_dir = f"{args.ckpt_dir}/entity/{args.dataset}/{args.model_name}"
    elif args.task == 'action prediction':
        args.dataset_dir = f"{args.processed_dir}/action/{args.dataset}"
        args.ckpt_dir = f"{args.ckpt_dir}/action/{args.dataset}/{args.model_name}"

    assert os.path.isdir(args.dataset_dir)

    with open(f"{args.dataset_dir}/{args.class_dict_name}.json", 'r') as f:
        class_dict = json.load(f)
    
    # For model & tokenizer setting
    print(f"Loading model & tokenizer for {args.mode}...")        
    tokenizer, model, args = load_model(args, class_dict=class_dict)
    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    model = model.to(args.device)
    model = DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
        
    vocab = tokenizer.get_vocab()
    args.bos_id = vocab[args.bos_token]
    args.eos_id = vocab[args.eos_token]
    args.pad_id = vocab[args.pad_token]
    args.speaker1_id = vocab[args.speaker1_token]
    args.speaker2_id = vocab[args.speaker2_token]
    
    print(f"Loading datasets & dataloaders for {args.task}...")
    if args.mode == 'train':
        if args.task == 'intent detection':
            train_set = IDDataset(args, args.train_prefix, class_dict, tokenizer)
            valid_set = IDDataset(args, args.valid_prefix, class_dict, tokenizer)

            loss_func = nn.CrossEntropyLoss()
        elif args.task == 'entity recognition':
            train_set = ERDataset(args, args.train_prefix, class_dict, tokenizer)
            valid_set = ERDataset(args, args.valid_prefix, class_dict, tokenizer)

            loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        elif args.task == 'action prediction':
            train_set = APDataset(args, args.train_prefix, class_dict, tokenizer)
            valid_set = APDataset(args, args.valid_prefix, class_dict, tokenizer)

            loss_func = nn.BCEWithLogitsLoss()

        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='max', 
            factor=args.factor, 
            patience=args.patience, 
            threshold=args.threshold
        )
        
        with open(f"{ckpt_dir}/{args.train_name}_excluded.pickle", 'wb') as f:
            pickle.dump(train_set.excluded, f)
        with open(f"{ckpt_dir}/{args.valid_name}_excluded.pickle", 'wb') as f:
            pickle.dump(valid_set.excluded, f)
            
        if args.gpus > 1:
            train_sampler = DistributedSampler(train_set, num_replicas=args.world_size, rank=args.rank)
        else:
            train_sampler = None

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
        writer = SummaryWriter(args.ckpt_dir)
        
        # Train functions
        if args.task == 'entity recognition':
            train_er(args, model, loss_func, optim, scheduler, train_loader, valid_loader, class_dict, writer)
        elif args.task == 'action prediction':
            train_ap(args, model, loss_func, optim, scheduler, train_loader, valid_loader, writer)
            
        writer.close()
        
    elif args.mode == 'test':
        if args.task == 'entity recognition':
            test_set = ERDataset(args, args.test_prefix, class_dict, tokenizer)
        elif args.task == 'action prediction':
            test_set = APDataset(args, args.test_prefix, class_dict, tokenizer)

        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        model.load_state_dict(torch.load(f"{args.ckpt_dir}/{args.ckpt_name}.ckpt"))
        
        # Test functions
        if args.task == 'entity recognition':
            _, test_scores = eval_er(args, model, test_loader, class_dict, loss_func=None)
        elif args.task == 'action prediction':
            _, test_scores = eval_ap(args, model, test_loader, loss_func=None)
            
        print("Test results")
        print(test_scores)
        

def train_er(args, model, loss_func, optim, scheduler, train_loader, valid_loader, class_dict, writer):
    print("Training with Entity Recognition.")
    best_crit = 0.0

    for epoch in range(1, args.num_epochs+1):
        print("#" * 50 + f"Epoch {epoch}" + "#"*50)
        model.train()

        train_losses = []
        train_preds = []
        train_trues = []

        for batch in tqdm(train_loader):      
            optim.zero_grad()
            batch_input_ids, batch_labels = batch
            batch_input_ids, batch_labels = \
                batch_input_ids.to(args.device), batch_labels.to(args.device)
            batch_masks = (batch_input_ids != args.pad_id).float()
            
            outputs = model(batch_input_ids, padding_masks=batch_masks)  # (B, L, C)
            
            loss = loss_func(outputs.view(-1, args.num_classes), batch_labels.view(-1))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()

            train_losses.append(loss.item())
            _, preds = torch.max(outputs, dim=-1)

            preds = preds.tolist()
            trues = batch_labels.tolist()
            true_labels = (batch_labels != -1).tolist()
            spots = [(label.index(True), len(label)-list(reversed(label)).index(True)) for label in true_labels]
            preds = [pred[spots[p][0]:spots[p][1]] for p, pred in enumerate(preds)]
            trues = [true[spots[t][0]:spots[t][1]] for t, true in enumerate(trues)]

            train_preds += preds
            train_trues += trues

        train_loss = np.mean(train_losses)
        print(f"Train loss: {train_loss}")
        
        train_scores = entity_scores(train_preds, train_trues, class_dict)
        train_log_list = make_log_list('Train', train_scores)
        print(" || ".join(train_log_list))
        
        if args.gpus == 1 or (args.rank % args.gpus == 0):
            print("Validation starts.")
            valid_loss, valid_scores = eval_er(args, model, valid_loader, class_dict, loss_func=loss_func)
            crit_metric = next(iter(valid_scores))
            valid_crit = valid_scores[crit_metric]

            scheduler.step(valid_crit)

            if valid_crit > best_crit:
                best_crit = valid_crit
                train_crit = train_scores[crit_metric]

                ckpt_name = f"{args.model_name}_entity_epoch{epoch}_{round(train_crit, 4)}_{round(valid_crit, 4)}.ckpt"
                torch.save(model.state_dict(), f"{args.ckpt_dir}/{ckpt_name}")
                print("*"*10 + "Current best model saved." + "*"*10)

            print(f"Valid loss: {valid_loss}")
            print(f"Best {crit_metric}: {best_crit}")
            valid_log_list = make_log_list('Valid', valid_scores)
            print(" || ".join(valid_log_list))

            write_summaries(writer, epoch, train_loss, valid_loss, train_scores, valid_scores)
            
    print(f"Training for Entity Recognition with {args.model_name} finished.")

        
def eval_er(args, model, eval_loader, class_dict, loss_func=None):
    print("Evaluating with Entity Recognition.")
    model.eval()

    eval_losses = []
    eval_preds = []
    eval_trues = []

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            batch_input_ids, batch_labels = batch
            batch_input_ids, batch_labels = \
                batch_input_ids.to(args.device), batch_labels.to(args.device)
            batch_masks = (batch_input_ids != args.pad_id).float()
            
            outputs = model(batch_input_ids, padding_masks=batch_masks)  # (B, L, C)

            if loss_func is not None:
                loss = loss_func(outputs.view(-1, args.num_classes), batch_labels.view(-1))  
                eval_losses.append(loss.item())

            _, preds = torch.max(outputs, dim=-1)

            preds = preds.tolist()
            trues = batch_labels.tolist()
            true_labels = (batch_labels != -1).tolist()
            spots = [(label.index(True), len(label)-list(reversed(label)).index(True)) for label in true_labels]
            preds = [pred[spots[p][0]:spots[p][1]] for p, pred in enumerate(preds)]
            trues = [true[spots[t][0]:spots[t][1]] for t, true in enumerate(trues)]
            
            eval_preds += preds
            eval_trues += trues

        eval_loss = np.mean(eval_losses) if len(eval_losses) > 0 else 0.0
        eval_scores = entity_scores(eval_preds, eval_trues, class_dict)

    return eval_loss, eval_scores        


def train_ap(args, model, loss_func, optim, scheduler, train_loader, valid_loader, writer):
    print("Training with Action Prediction.")
    best_crit = 0.0

    for epoch in range(1, args.num_epochs+1):
        print("#" * 50 + f"Epoch {epoch}" + "#"*50)
        model.train()

        train_losses = []
        train_preds = []
        train_trues = []

        for batch in tqdm(train_loader): 
            optim.zero_grad()

            batch_input_ids, batch_labels = batch
            batch_input_ids, batch_labels = \
                batch_input_ids.to(args.device), batch_labels.to(args.device)
            batch_masks = (batch_input_ids != args.pad_id).float()

            outputs = model(batch_input_ids, padding_masks=batch_masks)  # (B, C)

            loss = loss_func(outputs, batch_labels)  # ()

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()

            train_losses.append(loss.item())
            preds = (torch.sigmoid(outputs) > args.sigmoid_threshold).long()  # (B, C)
            train_preds += preds.tolist()
            train_trues += batch_labels.long().tolist()

        train_loss = np.mean(train_losses)
        print(f"Train loss: {train_loss}")

        train_scores = action_scores(train_preds, train_trues)
        train_log_list = make_log_list('Train', train_scores)
        print(" || ".join(train_log_list))

        if args.gpus == 1 or (args.rank % args.gpus == 0):
            print("Validation starts.")
            valid_loss, valid_scores = eval_ap(args, model, valid_loader, loss_func=loss_func)
            crit_metric = next(iter(valid_scores))
            valid_crit = valid_scores[crit_metric]
            scheduler.step(valid_crit)

            if valid_crit > best_crit:
                best_crit = valid_crit
                train_crit = train_scores[crit_metric]

                ckpt_name = f"{args.model_name}_intent_epoch{epoch}_{round(train_crit, 4)}_{round(valid_crit, 4)}.ckpt"
                torch.save(model.state_dict(), f"{args.ckpt_dir}/{ckpt_name}")
                print("*"*10 + "Current best model saved." + "*"*10)

            print(f"Valid loss: {valid_loss}")
            print(f"Best {crit_metric}: {best_crit}")
            valid_log_list = make_log_list('Valid', valid_scores)
            print(" || ".join(valid_log_list))

            write_summaries(writer, epoch, train_loss, valid_loss, train_scores, valid_scores)

    print(f"Training for Action Prediction with {args.model_name} finished.")

    
def eval_ap(args, model, eval_loader, loss_func=None):
    print("Evaluating with Action Prediction.")
    model.eval()

    eval_losses = []
    eval_preds = []
    eval_trues = []

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            batch_input_ids, batch_labels = batch
            batch_input_ids, batch_labels = \
                batch_input_ids.to(args.device), batch_labels.to(args.device)
            batch_masks = (batch_input_ids != args.pad_id).float()

            outputs = model(batch_input_ids, padding_masks=batch_masks)  # (B, C)

            if loss_func is not None:
                loss = loss_func(outputs, batch_labels)  # ()
                eval_losses.append(loss.item())
                
            preds = (torch.sigmoid(outputs) > args.sigmoid_threshold).long()  # (B, C)
            eval_preds += preds.tolist()
            eval_trues += batch_labels.long().tolist()

        eval_loss = np.mean(eval_losses) if len(eval_losses) > 0 else 0.0
        eval_scores = action_scores(eval_preds, eval_trues)

    return eval_loss, eval_scores

        
def make_log_list(prefix, scores):
    log_list = []
    for metric, score in scores.items():
        log_list.append(f"{prefix} {metric}: {score}")

    return log_list


def write_summaries(writer, epoch, train_loss, valid_loss, train_scores, valid_scores):
    writer.add_scalars(
        'losses', 
        {'Train': train_loss, 'Valid': valid_loss},
        epoch
    )
    for metric in train_scores:
        train_value = train_scores[metric]
        valid_value = valid_scores[metric]

        writer.add_scalars(
            f'{metric}s', 
            {'Train': train_value, 'Valid': valid_value},
            epoch
        )


if __name__=='__main__':
    assert torch.cuda.is_available(), "Only GPU available environment is supported."
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', required=True, type=str, help="train/test?")
    parser.add_argument('--task', required=True, type=str, help="The name of finetune task.")
    parser.add_argument('--dataset', required=True, type=str, help="The name of dataset name.")
    parser.add_argument('--model_name', required=True, type=str, help="The name of pre-trained model type.")
    parser.add_argument('--ckpt_dir', required=True, type=str, default="saved_models", help="The directory path for saved checkpoints.")
    parser.add_argument('--ckpt_name', required=False, type=str, help="Checkpoint file name.")
    parser.add_argument('--data_dir', required=True, type=str, default="data", help="The parent directory path for data files.")
    parser.add_argument('--processed_dir', required=True, type=str, default="processed", help="The directory path to finetuning data files.")
    parser.add_argument('--class_dict_name', required=True, type=str, default="class_dict", help="The name of class dictionary json file.")
    parser.add_argument('--train_prefix', required=True, type=str, default="train", help="The prefix of file name related to train set.")
    parser.add_argument('--valid_prefix', required=True, type=str, default="valid", help="The prefix of file name related to valid set.")
    parser.add_argument('--test_prefix', required=True, type=str, default="test", help="The prefix of file name related to test set.")
    parser.add_argument('--utter_name', required=True, type=str, default="utter", help="The indication for utterance files' name.")
    parser.add_argument('--label_name', required=True, type=str, default="label", help="The indication for label files' name.")
    parser.add_argument('--entity_dir', required=True, type=str, default="entity", help="The directory path for entity recognition data files.")
    parser.add_argument('--action_dir', required=True, type=str, default="action", help="The directory path for action prediction data files.")
    parser.add_argument('--max_len', required=True, type=int, default=512, help="The maximum sequence length including all dialog contexts.")
    parser.add_argument('--max_times', required=True, type=int, default=5, help="The maximum number of time steps.")
    parser.add_argument('--cuda',  action='store_true', help="The flag for device setting, cuda or not?")
    parser.add_argument('--num_epochs', required=True, type=int, default=50, help="The number of total epochs.")
    parser.add_argument('--batch_size', required=True, type=int, default=32, help="The batch size in one process.")
    parser.add_argument('--num_workers', required=True, type=int, default=1, help="The number of workers for data loading.")
    parser.add_argument('--learning_rate', required=True, type=float, default=5e-5, help="The starting learning rate.")
    parser.add_argument('--factor', required=True, type=float, default=0.1, help="The scheduling factor.")
    parser.add_argument('--max_grad_norm', required=True, type=float, default=1.0, help="The max gradient for gradient clipping.")
    parser.add_argument('--patience', required=True, type=int, default=3, help="The patience epoch.")
    parser.add_argument('--threshold', required=True, type=float, default=1e-3, help="The threshold to decide learning rate scheduling.")
    parser.add_argument('--sigmoid_threshold', required=True, type=float, default=0.5, help="The sigmoid threshold for action prediction task.")
    
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    
    args = parser.parse_args()
        
    assert args.mode == 'train' or args.mode == 'test', "Please specify a correct mode."
    assert args.task == 'entity recognition'\
        or args.task == 'action prediction', \
        "You must specify a correct finetune task in finetuning mode."
    assert args.model_name is not None, "You must specify the pre-trained model."
    assert args.dataset is not None, "You must specify the dataset name."
    
    print("#"*50 + "Running spec" + "#"*50)
    print(f"Mode: {args.mode}")
    print(f"Task: {args.task}")
    print(f"Dataset: {args.dataset}")
    print(f"Model name: {args.model_name}")
    print(f"Checkpoint name: {args.ckpt_name}")
    print(f"Number of GPUs: {args.gpus}")
    
    input("Press Enter to continue...")
    
    args.world_size = args.gpus * args.nodes
    
    mp.spawn(main, nprocs=args.gpus, args=(args, ))
