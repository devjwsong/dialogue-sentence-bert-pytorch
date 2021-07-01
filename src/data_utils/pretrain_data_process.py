from pretrain import schema, frames, e2e

import argparse
import os


data_list = ["schema", "frames", "e2e"]


def process_data(args, pretrain_dir):
    print("Processing data for pretraining...")
    if not os.path.isdir(pretrain_dir):
        os.makedirs(pretrain_dir)
    
    for data_name in data_list:
        print("#" * 100)
        print(f"Processing {data_name}...")
        
        if data_name == 'schema':
            schema.process_data(args, pretrain_dir)
        elif data_name == 'frames':
            frames.process_data(args, pretrain_dir)
        elif data_name == 'e2e':
            e2e.process_data(args, pretrain_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="data", help="The parent directory path for data files.")
    parser.add_argument('--raw_dir', type=str, default="raw", help="The directory path for raw data files.")
    parser.add_argument('--pretrain_dir', type=str, default="pretrain", help="The directory path to processed pretrain data files.")
    parser.add_argument('--train_frac', type=float, default=0.8, help="The ratio of the conversations to be included in the train set.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of file name related to train set.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of file name related to valid set.")
    parser.add_argument('--action_map_name', type=str, default="action_map", help="The name of action map json file.")
    
    args = parser.parse_args()
    
    pretrain_dir = f"{args.data_dir}/{args.pretrain_dir}"
    process_data(args, pretrain_dir)
