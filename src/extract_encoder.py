from pretrain_module import pt_module

import torch
import argparse
import os


def extract(args):
    ckpt_dir = f"{args.default_root_dir}/lightning_logs/version_{args.log_idx}/checkpoints"
    pt_module = PretrainModule.load_from_checkpoint(f"{ckpt_dir}/{args.ckpt_file}")
    
    tokenizer = pt_module.tokenizer
    model = pt_module.encoder
    config = model.config
    
    print("CHECK!")
    print(config)
    print(tokenizer)
    print(model)
    
    pooling = pt_module.args.pooling
    output_dir = f"{ckpt_dir}/{args.ckpt_file.split('.ckpt')[0]}"
    
    config.save_pretrained(output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--default_root_dir', type=str, default="./", help="The default directory for logs & checkpoints.")
    parser.add_argument('--log_idx', type=int, required=True, help="The lightning log index.")
    parser.add_argument('--ckpt_file', type=str, required=True, help="The checkpoint file name to extract.")
    
    args = parser.parse_args()
    
    extract(args)
    
    print("FINISHED.")
