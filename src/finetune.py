from torch.utils.data import DataLoader
from model_utils.finetune_module import *
from data_utils.finetune_datasets import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from utils import convert_gpu_str_to_list

import os
import argparse
import json


def run(args):
    # For directory setting
    args.dataset_dir = f"{args.data_dir}/{args.finetune_dir}/{args.dataset}"
    assert os.path.isdir(args.dataset_dir)
    
    args.cache_dir = f"{args.cache_dir}/{args.task}/{args.dataset}/{args.model_name}"
    if args.task == 'action':
        args.cache_dir = f"{args.cache_dir}/{args.max_turns}"
    
    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir)

    with open(f"{args.dataset_dir}/{args.task}_{args.class_dict_name}.json", 'r') as f:
        args.class_dict = json.load(f)
    args.num_classes = len(args.class_dict)
    
    print(f"Loading training module for {args.task}...")   
    module = FinetuneModule(args)
    args = module.args
    
    print("Loading datasets...")
    # For data loading
    if args.task == 'intent':
        train_set = IDDataset(args, args.train_prefix, module.tokenizer)
        valid_set = IDDataset(args, args.valid_prefix, module.tokenizer)
        test_set = IDDataset(args, args.test_prefix, module.tokenizer)
    elif args.task == 'action':
        train_set = APDataset(args, args.train_prefix, module.tokenizer)
        valid_set = APDataset(args, args.valid_prefix, module.tokenizer)
        test_set = APDataset(args, args.test_prefix, module.tokenizer)

    # Dataloaders
    input_pad_id = args.pad_id
    ppd = PadCollate(input_pad_id=input_pad_id)
    
    # Reset random seed for data shuffle
    train_loader = DataLoader(train_set, collate_fn=ppd.pad_collate, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, collate_fn=ppd.pad_collate, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, collate_fn=ppd.pad_collate, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    # Calculate total training steps
    num_batches = len(train_loader)
    args.total_train_steps = args.num_epochs * num_batches
    args.warmup_steps = int(args.warmup_prop * args.total_train_steps)

    print("Setting pytorch lightning callback & trainer...")
    # Model checkpoint callback
    if args.task == 'intent':
        filename = "best_ckpt_{epoch}_{train_all_acc:.4f}_{valid_all_acc:.4f}"
        monitor = "valid_all_acc"
    elif args.task == 'action':
        filename = "best_ckpt_{epoch}_{train_samples_f1:.4f}_{valid_samples_f1:.4f}"
        monitor = "valid_samples_f1"
    
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        verbose=True,
        monitor=monitor,
        mode='max',
        every_n_epochs=1,
        save_weights_only=True
    )
    stopping_callback = EarlyStopping(
        monitor=monitor,
        min_delta=1e-4,
        patience=3,
        verbose=True,
        mode='max'
    )
    
    # Trainer setting
    args.gpu = convert_gpu_str_to_list(args.gpu)
    trainer = Trainer(
        accelerator="gpu",
        gpus=args.gpu,
        check_val_every_n_epoch=1,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.max_grad_norm,
        deterministic=True,
        callbacks=[checkpoint_callback, stopping_callback]
    )
    
    print("Train starts.")
    seed_everything(args.seed, workers=True)
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    print("Training done.")
    
    print("Test starts.")
    trainer.test(dataloaders=test_loader, ckpt_path='best')
    
    print("GOOD BYE.")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--task', required=True, type=str, help="The name of the task.")
    parser.add_argument('--dataset', required=True, type=str, help="The name of the dataset.")
    parser.add_argument('--cache_dir', type=str, default="cached", help="The directory path for pre-processed data pickles.")
    parser.add_argument('--data_dir', type=str, default="data", help="The parent directory path for data files.")
    parser.add_argument('--finetune_dir', type=str, default="finetune", help="The directory path to finetuning data files.")
    parser.add_argument('--class_dict_name', type=str, default="class_dict", help="The name of class dictionary json file.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of file name related to train set.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of file name related to valid set.")
    parser.add_argument('--test_prefix', type=str, default="test", help="The prefix of file name related to test set.")
    parser.add_argument('--max_turns', type=int, default=1, help="The maximum number of dialogue turns.")
    parser.add_argument('--num_epochs', type=int, default=1, help="The number of total epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="The batch size in one process.")
    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument('--max_encoder_len', type=int, default=512, help="The maximum length of a sequence.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="The starting learning rate.")
    parser.add_argument('--warmup_prop', type=float, default=0.0, help="The warmup step proportion.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The max gradient for gradient clipping.")
    parser.add_argument('--sigmoid_threshold', type=float, default=0.5, help="The sigmoid threshold for action prediction task.")
    parser.add_argument('--cached', action="store_true", help="Using the cached data or not?")
    parser.add_argument('--seed', type=int, default=0, help="The seed number.")
    parser.add_argument('--model_name', required=True, type=str, help="The encoder model to test.")
    parser.add_argument('--pooling', required=True, type=str, help="Pooling method: CLS/Mean/Max")
    parser.add_argument('--gpu', type=str, default="0", help="The index of GPU to use.")
    parser.add_argument('--ckpt_dir', required=False, type=str, help="If only training from a specific checkpoint... (also convbert)")
    parser.add_argument('--ckpt_name', required=False, type=str, help="If only training from a specific checkpoint...")
    
    args = parser.parse_args()
    
    assert args.task == 'intent' or args.task == 'action', "You must specify a correct dialogue task."
    assert args.model_name in [
        'bert',  'convbert', 'todbert', 'sentbert-cls', 'sentbert-mean', 'sentbert-max',
        'dialogsentbert-cls', 'dialogsentbert-mean', 'dialogsentbert-max',
        'dialogsentconvbert-cls', 'dialogsentconvbert-mean', 'dialogsentconvbert-max',
        'dialogsenttodbert-cls', 'dialogsenttodbert-mean', 'dialogsenttodbert-max',
    ], "You must specify a correct model name."
    assert args.pooling in ['cls', 'mean', 'max']
    if 'sent' in args.model_name:
        assert args.model_name.split('-')[-1] == args.pooling
    if 'conv' in args.model_name:
        assert args.ckpt_dir is not None
    if 'dialogsent' in args.model_name:
        assert args.ckpt_dir is not None and args.ckpt_name is not None
    
    print("#"*50 + "Running spec" + "#"*50)
    print(args)
    
    input("Please press Enter to proceed...")
    
    run(args)
