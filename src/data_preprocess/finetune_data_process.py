from finetune import oos, banking77, atis, multiwoz, dstc2, sim

import argparse
import os


data_list = ["oos", "banking77", "atis", "multiwoz", "dstc2", "sim"]
            

def process_data(args):
    print("Processing data for finetuning...")
    if not os.path.isdir(args.finetune_dir):
        os.makedirs(args.finetune_dir)
    
    for data_name in data_list:
        print("#" * 100)
        print(f"Processing {data_name}...")
        
        if data_name == 'oos':
            oos.process_data(args)
        elif data_name == 'banking77':
            banking77.process_data(args)
        elif data_name == 'atis':
            atis.process_data(args)
        elif data_name == 'multiwoz':
            multiwoz.process_data(args)
        elif data_name == 'dstc2':
            dstc2.process_data(args)
        elif data_name == 'sim':
            sim.process_data(args)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_dir', type=str, default="data/raw", help="The directory path for raw data files.")
    parser.add_argument('--finetune_dir', type=str, default="data/finetune", help="The directory path to processed finetune data files.")
    parser.add_argument('--train_frac', type=float, default=0.8, help="The ratio of the conversations to be included in the train set.")
    parser.add_argument('--valid_frac', type=float, default=0.1, help="The ratio of the conversations to be included in the valid set.")
    parser.add_argument('--train_prefix', type=str, default="train", help="The prefix of file name related to train set.")
    parser.add_argument('--valid_prefix', type=str, default="valid", help="The prefix of file name related to valid set.")
    parser.add_argument('--test_prefix', type=str, default="test", help="The prefix of file name related to test set.")
    parser.add_argument('--class_dict_name', type=str, default="class_dict", help="The name of class dictionary json file.")
    
    args = parser.parse_args()
    
    process_data(args)
