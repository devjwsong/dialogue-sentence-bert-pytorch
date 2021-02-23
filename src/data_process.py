from tqdm import tqdm
from data_utils import multiwoz, taskmaster3, dstc2, sim

import argparse
import os


# data_list = ["multiwoz", "taskmaster3", "dstc2", "sim"]
data_list = ['multiwoz']


def process_data(args, processed_dir):
    print("Processing data for tasks...")
    if not os.path.isdir(processed_dir):
        os.mkdir(processed_dir)
    
    for data_name in data_list:
        print("#" * 100)
        print(f"Processing {data_name}...")
        
        if data_name == 'multiwoz':
            multiwoz.process_data(args, processed_dir)
        elif data_name == 'taskmaster3':
            taskmaster3.process_data(args, processed_dir)
        elif data_name == 'dstc2':
            dstc2.process_data(args, processed_dir)
        elif data_name == 'sim':
            sim.process_data(args, processed_dir)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, type=str, default="data", help="The parent directory path for data files.")
    parser.add_argument('--raw_dir', required=True, type=str, default="raw", help="The directory path for raw data files.")
    parser.add_argument('--processed_dir', required=True, type=str, default="processed", help="The directory path to processed data files.")
    parser.add_argument('--entity_dir', required=True, type=str, default="entity", help="The directory path for entity recognition data files.")
    parser.add_argument('--action_dir', required=True, type=str, default="action", help="The directory path for action prediction data files.")
    parser.add_argument('--train_frac', required=True, type=float, default=0.8, help="The ratio of the conversations to be included in the train set.")
    parser.add_argument('--valid_frac', required=True, type=float, default=0.1, help="The ratio of the conversations to be included in the valid set.")
    parser.add_argument('--train_prefix', required=True, type=str, default="train", help="The prefix of file name related to train set.")
    parser.add_argument('--valid_prefix', required=True, type=str, default="valid", help="The prefix of file name related to valid set.")
    parser.add_argument('--test_prefix', required=True, type=str, default="test", help="The prefix of file name related to test set.")
    parser.add_argument('--utter_name', required=True, type=str, default="utter", help="The indication for utterance files' name.")
    parser.add_argument('--label_name', required=True, type=str, default="label", help="The indication for label files' name.")
    parser.add_argument('--class_dict_name', required=True, type=str, default="class_dict", help="The name of class dictionary json file.")
    
    args = parser.parse_args()
    
    processed_dir = f"{args.data_dir}/{args.processed_dir}"
    
    process_data(args, processed_dir)
