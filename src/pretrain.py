from pretrain_module import *
from pytorch_lightning import Trainer, seed_everything
from utils import convert_gpu_str_to_list

import argparse


def run(args):
    args.gpus = convert_gpu_str_to_list(args.gpus)
    
    print(f"Loading training module for pretraining...")   
    module = PretrainModule(args)
    args = module.args

    print("Setting pytorch lightning callback & trainer...")
    # Model checkpoint callback
    checkpoint_callback = CustomModelCheckpoint(
        every_n_train_steps=args.save_interval,
        save_weights_only=True,
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
        accelerator="ddp",
        checkpoint_callback=False,
        amp_backend="apex",
        amp_level=args.amp_level,
        replace_sampler_ddp=False,
    )
    
    print("Train starts.")
    trainer.fit(model=module)
    print("Training done.")
    
    print("GOOD BYE.")
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--default_root_dir', type=str, default="./", help="The default directory for logs & checkpoints.")
    parser.add_argument('--shuffled_dir', type=str, default="pretrain-shuffled", help="The directory which contains the shuffled pre-train data files.")
    parser.add_argument('--num_epochs', type=int, default=1, help="The number of total epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="The batch size assigned to each GPU.")
    parser.add_argument('--num_workers', type=int, default=4, help="The number of workers for data loading.")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="The initial learning rate.")
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help="The warmup step ratio.")
    parser.add_argument('--save_interval', type=int, default=50000, help="The training step interval to save checkpoints.")
    parser.add_argument('--log_interval', type=int, default=10000, help="The training step interval to write logs.")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The max gradient for gradient clipping.")
    parser.add_argument('--seed', type=int, default=0, help="The random seed number.")
    parser.add_argument('--pooling', type=str, required=True, help="Pooling method: CLS/Mean/Max.")
    parser.add_argument('--ckpt_dir', required=False, type=str, help="If only training from a specific checkpoint... (also convbert)")
    parser.add_argument('--gpus', type=str, default="0, 1, 2, 3", help="The indices of GPUs to use.")
    parser.add_argument('--amp_level', type=str, default="O1", help="The optimization level to use for 16-bit GPU precision.")
    parser.add_argument('--num_nodes', type=int, default=1, help="The number of machine.")
    
    args = parser.parse_args()
    
    args.model_name = "bert"
    assert args.pooling in ['cls', 'mean', 'max'], "You must specify a correct pooling method."
    
    run(args)
