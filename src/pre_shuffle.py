from glob import glob
from tqdm import tqdm

import natsort
import argparse
import random
import ujson
import os


def run(args):
    first_sample_files = glob(f"{args.data_dir}/{args.pretrain_dir}/first*")
    second_sample_files = glob(f"{args.data_dir}/{args.pretrain_dir}/second*")
    label_files = glob(f"{args.data_dir}/{args.pretrain_dir}/label*")
    
    assert len(first_sample_files) == len(second_sample_files)
    assert len(first_sample_files) == len(label_files)
    
    first_sample_files = natsort.natsorted(first_sample_files)
    second_sample_files = natsort.natsorted(second_sample_files)
    label_files = natsort.natsorted(label_files)
    
    cur_group = 0
    for i in tqdm(range(0, len(first_sample_files), args.num_files)):
        target_first_sample_files = first_sample_files[i:i+args.num_files]
        target_second_sample_files = second_sample_files[i:i+args.num_files]
        target_label_files = label_files[i:i+args.num_files]
        
        samples = []
        num_samples = 0
        for j in range(len(target_first_sample_files)):
            with open(target_first_sample_files[j], 'r') as f:
                first_samples = ujson.load(f)
            with open(target_second_sample_files[j], 'r') as f:
                second_samples = ujson.load(f)
            with open(target_label_files[j], 'r') as f:
                label_samples = ujson.load(f)
                
            assert len(first_samples) == len(second_samples)
            assert len(first_samples) == len(label_samples)
            num_samples += len(first_samples)
            
            if i == 0 and j == 0:
                args.group_size = len(first_samples)
                
            for k in range(len(first_samples)):
                sample = (first_samples[k], second_samples[k], label_samples[k])
                samples.append(sample)
                
        assert num_samples == len(samples)
                
        random.shuffle(samples)
        
        for j in range(len(target_first_sample_files)):
            first_samples, second_samples, labels = zip(*samples[j*args.group_size:(j+1)*args.group_size])
            with open(f"{args.save_dir}/first_samples_group{cur_group}.json", 'w') as f:
                ujson.dump(first_samples, f)
            with open(f"{args.save_dir}/second_samples_group{cur_group}.json", 'w') as f:
                ujson.dump(second_samples, f)
            with open(f"{args.save_dir}/label_group{cur_group}.json", 'w') as f:
                ujson.dump(labels, f)
            cur_group += 1


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help="The random seed number.")
    parser.add_argument('--data_dir', type=str, default="data", help="The parent directory path for data files.")
    parser.add_argument('--pretrain_dir', type=str, default="pretrain", help="The directory which contains the pre-train data files.")
    parser.add_argument('--num_files', type=int, default=1, help="The number of files to be shuffled together.")
    
    args = parser.parse_args()
    
    args.save_dir = f"{args.data_dir}/{args.pretrain_dir}_shuffled"
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    random.seed(args.seed)
    
    run(args)
