from torch.utils.data import DataLoader
from model_utils.pretrain_module import *
from data_utils.pretrain_datasets import *
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

import os
import argparse
import random


def run(args):
    # For directory setting
    args.pretrain_dir = f"{args.data_dir}/{args.pretrain_dir}"
    assert os.path.isdir(args.pretrain_dir)
    
    args.cached_dir = f"{args.cached_dir}/pretrain/{args.model_name}"
    if not os.path.isdir(args.cached_dir):
        os.makedirs(args.cached_dir)

    class_dict = {
        "same": 0,
        "diff": 1
    }
    args.num_classes = len(class_dict)
    
    print(f"Loading training module for pretraining...")   
    module = PretrainModule(args)
    args = module.args
    
    print("Loading datasets...")
    # For data loading
    train_set = PretrainDataset(args, args.train_prefix, module.tokenizer, class_dict)
    valid_set = PretrainDataset(args, args.valid_prefix, module.tokenizer, class_dict)

    # Dataloaders
    input_pad_id = args.pad_id
    ppd = PretrainPadCollate(input_pad_id=input_pad_id)
    
    # Reset random seed for data shuffle
    seed_everything(args.seed, workers=True)

    train_loader = DataLoader(train_set, collate_fn=ppd.pad_collate, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, collate_fn=ppd.pad_collate, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    # Calculate total training steps
    num_gpus = len(args.gpus.split(' '))
    num_devices = num_gpus * args.num_nodes
    q, r = divmod(len(train_loader), num_devices)
    num_batches = q if r == 0 else q+1
    args.total_train_steps = args.num_epochs * num_batches
    args.warmup_steps = int(args.warmup_prop * args.total_train_steps)

    print("Setting pytorch lightning callback & trainer...")
    # Model checkpoint callback
    filename = f"{args.model_name}_{args.pooling}" + "_{epoch}_{train_acc:.4f}_{valid_acc:.4f}"
    monitor = "valid_acc"
    
    checkpoint_callback = ModelCheckpoint(
        filename=filename,
        verbose=True,
        monitor=monitor,
        mode='max',
        every_n_val_epochs=1,
        save_weights_only=True
    )
    
    # Trainer setting
    trainer = Trainer(
        check_val_every_n_epoch=1,
        gpus=args.gpus,
        auto_select_gpus=True,
        num_nodes=args.num_nodes,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.max_grad_norm,
        num_sanity_val_steps=0,
        deterministic=True,
        callbacks=[checkpoint_callback],
        accelerator="ddp"
    )
    
    print("Train starts.")
    trainer.fit(model=module, train_dataloader=train_loader, val_dataloaders=valid_loader)
    print("Training done.")
    
    print("GOOD BYE.")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default="data", help="The parent directory path for data files.")
    parser.add_argument('--pretrain_dir', type=str, default="pretrain", help="The directory path to pretraining data files.")
    parser.add_argument('--cached_dir', type=str, default="cached", help="The directory path for pre-processed data pickles.")
    parser.add_argument('--cached', action="store_true", help="Using the cached data or not?")
    parser.add_argument('--input_prefix', type=str, default="input", help="The prefix of file name related to input files.")
    parser.add_argument('--label_prefix', type=str, default="label", help="The prefix of file name related to label files.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of file name related to train set.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of file name related to valid set.")
    parser.add_argument('--max_turns', type=int, default=1, help="The maximum number of dialogue contexts.")
    parser.add_argument('--num_epochs', type=int, default=1, help="The number of total epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="The batch size in one process.")
    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument('--max_encoder_len', type=int, default=512, help="The maximum length of a sequence.")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="The starting learning rate.")
    parser.add_argument('--warmup_prop', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The max gradient for gradient clipping.")
    parser.add_argument('--seed', type=int, default=0, help="The seed number.")
    parser.add_argument('--model_name', required=True, type=str, help="The encoder model to train.")
    parser.add_argument('--pooling', required=True, type=str, help="Pooling method: CLS/Mean/Max")
    parser.add_argument('--ckpt_dir', required=False, type=str, help="If only training from a specific checkpoint...")
    parser.add_argument('--ckpt_name', required=False, type=str, help="If only training from a specific checkpoint...")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--num_nodes', type=int, default=1)
    parser.add_argument('--group_size', type=int, default=1000)
    parser.add_argument('--num_train_samples', type=int, default=1000000)
    parser.add_argument('--num_valid_samples', type=int, default=1000000)
    
    args = parser.parse_args()
    
    assert args.model_name in ['bert', 'convert', 'todbert'], "You must specify a correct model name."
    assert args.pooling in ['cls', 'mean', 'max'], "You must specify a correct pooling method."
    if 'conv' in args.model_name:
        assert args.ckpt_dir is not None
    
    print("#"*50 + "Running spec" + "#"*50)
    print(args)
    
    input("Please press Enter to proceed...")
    
    run(args)
