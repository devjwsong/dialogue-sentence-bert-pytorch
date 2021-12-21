from torch.utils.data import DataLoader
from model_utils.pretrain_module import *
from data_utils.pretrain_datasets import *
from pytorch_lightning import Trainer, seed_everything
from utils import convert_gpu_str_to_list

import argparse


def run(args):
    print(f"Loading training module for pretraining...")   
    module = PretrainModule(args)
    args = module.args
    
    print("Loading datasets...")
    # For data loading
    train_set = PretrainDataset(args)

    # Dataloaders
    input_pad_id = args.pad_id
    ppd = PretrainPadCollate(input_pad_id=input_pad_id)
    
    # Reset random seed for data shuffle
    train_loader = DataLoader(train_set, collate_fn=ppd.pad_collate, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    # Calculate total training steps
    args.gpus = convert_gpu_str_to_list(args.gpus)
    num_devices = len(args.gpus) * args.num_nodes
    q, r = divmod(len(train_loader), num_devices)
    num_batches = q if r == 0 else q+1
    args.num_training_steps = args.num_epochs * num_batches
    args.num_warmup_steps = int(args.warmup_ratio * args.num_training_steps)

    print("Setting pytorch lightning callback & trainer...")
    # Model checkpoint callback
    checkpoint_callback = CustomModelCheckpoint(
        every_n_train_steps=args.save_interval,
        save_weights_only=True,
        num_steps_per_epoch=num_batches,
    )
    
    # Trainer setting
    seed_everything(args.seed, workers=True)
    trainer = Trainer(
        default_root_dir=args.default_root_dir,
        gpus=args.gpus,
        num_nodes=args.num_nodes,
        max_epochs=args.num_epochs,
        gradient_clip_val=args.max_grad_norm,
        log_every_n_steps=args.log_interval,
        num_sanity_val_steps=0,
        deterministic=True,
        callbacks=[checkpoint_callback],
        strategy="ddp",
        enable_checkpointing=False,
        amp_backend="apex",
        amp_level=args.amp_level,
    )
    
    print("Train starts.")
    trainer.fit(model=module, train_dataloaders=train_loader)
    print("Training done.")
    
    print("GOOD BYE.")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--default_root_dir', type=str, default="./", help="The default directory for logs & checkpoints.")
    parser.add_argument('--data_dir', type=str, default="data", help="The parent directory path for data files.")
    parser.add_argument('--pretrain_dir', type=str, default="data/pretrain", help="The directory which contains the pre-train data files.")
    parser.add_argument('--num_epochs', type=int, default=1, help="The number of total epochs.")
    parser.add_argument('--batch_size', type=int, default=16, help="The batch size in one process.")
    parser.add_argument('--num_workers', type=int, default=0, help="The number of workers for data loading.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="The starting learning rate.")
    parser.add_argument('--warmup_ratio', type=float, default=0.0, help="The warmup step ratio.")
    parser.add_argument('--save_interval', type=int, default=50000, help="The training step interval to save checkpoints.")
    parser.add_argument('--log_interval', type=int, default=10000, help="The training step interval to write logs.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The max gradient for gradient clipping.")
    parser.add_argument('--seed', type=int, default=0, help="The random seed number.")
    parser.add_argument('--pooling', required=True, type=str, help="Pooling method: CLS/Mean/Max")
    parser.add_argument('--ckpt_dir', required=False, type=str, help="If only training from a specific checkpoint... (also convbert)")
    parser.add_argument('--gpus', type=str, default="0", help="The indices of GPUs to use.")
    parser.add_argument('--amp_level', type=str, default="O1", help="The optimization level to use for 16-bit GPU precision.")
    parser.add_argument('--num_nodes', type=int, default=1, help="The number of machine.")
    
    args = parser.parse_args()
    
    args.model_name = "bert"
    assert args.pooling in ['cls', 'mean', 'max'], "You must specify a correct pooling method."
    
    run(args)
