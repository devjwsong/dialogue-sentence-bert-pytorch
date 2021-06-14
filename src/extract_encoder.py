from model_utils.pretrain_module import *

import torch
import argparse
import os


def extract(args):
    ckpt_dir = f"lightning_logs/version_{args.log_idx}/checkpoints"
    pt_module = PretrainModule.load_from_checkpoint(f"{ckpt_dir}/{args.ckpt_name}.ckpt")
    
    encoder = pt_module.encoder
    
    print("CHECK!")
    print(f"Encoder class: {type(encoder)}")
    
    model_name, pooling = args.ckpt_name.split("_")[:2]
    args.save_dir = f"{args.save_dir}/dialogsent{model_name}-{pooling}"
    
    torch.save(encoder.state_dict(), f"{args.save_dir}/{args.ckpt_name}.pt")


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--log_idx', type=int, required=True)
    parser.add_argument('--ckpt_name', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default="saved_models")
    
    args = parser.parse_args()
    
    extract(args)
    
    print("FINISHED.")
