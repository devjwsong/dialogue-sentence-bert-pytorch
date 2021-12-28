from tqdm import tqdm
from glob import glob

import argparse
import json
import random
import numpy as np
import os


def save_files(args, sample_list_0, sample_list_1, labels, cur_group):
    assert len(sample_list_0) == len(sample_list_1)
    assert len(sample_list_0) == len(labels)
    
    with open(f"{args.save_dir}/sample_list_0_group_{cur_group}.pkl", 'w') as f:
        json.dump(sample_list_0, f)
    with open(f"{args.save_dir}/sample_list_1_group_{cur_group}.pkl", 'w') as f:
        json.dump(sample_list_1, f)
    with open(f"{args.save_dir}/labels_group_{cur_group}.pkl", 'w') as f:
        json.dump(labels, f)

        
def sample_negative(valid_idxs, keys):
    key_idxs = list(range(len(keys)))
    key_idx = random.sample(key_idxs, 1)[0]
    sampled_file = keys[key_idx]
    with open(sampled_file, 'rb) as f:
        sampled = json.load(f)
        
    negative = sampled[valid_idxs[sampled_file]]
    valid_idxs[sampled_file] += 1
    if valid_idxs[sampled_file] >= len(sampled):
        valid_idxs.pop(sampled_file, None)
        keys = keys[:key_idx] + keys[key_idx+1:]
        
    assert len(keys) == len(valid_idxs)-1
        
    return negative, valid_idxs, keys


def get_diff_pair(diff0, valid_idxs, keys):
    sample_list_0.append(diff0)
    diff1, valid_idxs, keys = sample_negative(valid_idxs, keys)
    sample_list_1.append(diff1)
    labels.append(1)
    lens.append(len(diff0))
    lens.append(len(diff1))
    
    return valid_idxs, keys
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=0, help="The random seed.")
    parser.add_argument('--parsed_dir', type=str, default="data/opensubtitles-parsed", help="The parent directory for saving parsed data.")
    parser.add_argument('--keep_ratio', type=float, default="0.66", help="The ratio of sampled to be kept as the same pairs.")
    parser.add_argument('--save_dir', type=str, default="data/pretrain", help="The directory for pre-train data files.")
    parser.add_argument('--group_size', type=int, default=100000, help="The maximum number of samples in each file.")

    args = parser.parse_args()

    random.seed(args.seed)
    
    file_list = glob(f"{args.parsed_dir}/*.pickle")
    print(f"The number of files: {len(file_list)}.")
    
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)
    
    valid_idxs = {file: 0 for file in file_list}
    sample_list_0, sample_list_1, labels = [], [], []
    cur_group = 0
    lens, num_same, num_diff = [], 0, 0
    
    for file in tqdm(file_list):
        with open(file, 'r') as f:
            seqs = json.load(f)
        
        if file not in valid_idxs:
            continue
            
        cur_idx = valid_idxs[file]
        seqs = seqs[cur_idx:]
        keys = [k for k, v in valid_idxs.items() if k != file]
        
        num_kept = int(len(seqs) * args.keep_ratio)
        if num_kept % 2 == 1:
            num_kept += 1
            
        diff_seqs = []
        if num_kept > 0:
            idxs = list(range(len(seqs)))
            sampled_idxs = random.sample(idxs, num_kept)
            sampled_idxs = sorted(sampled_idxs)

            prev_idx = sampled_idxs[0]
            diff_seqs += seqs[:prev_idx]
            for idx in sampled_idxs[1:]:
                diff_seqs += seqs[prev_idx+1:idx]
                prev_idx = idx
            diff_seqs += seqs[sampled_idxs[-1]+1:]

            for i in range(0, len(sampled_idxs), 2):
                idx0, idx1 = sampled_idxs[i], sampled_idxs[i+1]
                sample_list_0.append(seqs[idx0])
                sample_list_1.append(seqs[idx1])
                labels.append(0)
                num_same += 1
                lens.append(len(seqs[idx0]))
                lens.append(len(seqs[idx1]))
        else:
            if len(seqs) == 1:
                diff_seqs = seqs
                
        count = 0
        for seq in diff_seqs:
            if len(keys) == 0:
                break
            valid_idxs, keys = get_diff_pair(seq, valid_idxs, keys)
            count += 1
            num_diff += 1
            
        if count < len(diff_seqs) and len(diff_seqs) - count > 1:
            extra = diff_seqs[count:]
            random.shuffle(extra)
            if len(extra) % 2 == 1:
                extra = extra[:-1]
            for i in range(0, len(extra), 2):
                sample_list_0.append(extra[i])
                sample_list_1.append(extra[i+1])
                labels.append(0)
                num_same += 1
                lens.append(len(extra[i]))
                lens.append(len(extra[i+1]))
                
        if len(sample_list_0) >= args.group_size:
            save_files(
                args,
                sample_list_0[:args.group_size],
                sample_list_1[:args.group_size],
                labels[:args.group_size],
                cur_group
            )
            sample_list_0 = sample_list_0[args.group_size:]
            sample_list_1 = sample_list_1[args.group_size:]
            labels = labels[args.group_size:]
            cur_group += 1

        valid_idxs.pop(file, None)

        if len(valid_idxs) == 0:
            break
            
    if len(sample_list_0) > 0:
        save_files(args, sample_list_0, sample_list_1, labels, cur_group)
    
    assert 2 * (num_same + num_diff) == len(lens)
    
    print(f"The number of same pairs: {num_same}.")
    print(f"The number of diff pairs: {num_diff}.")
    print(f"The number of total sequences: {len(lens)}")
    print(f"The maximum input length: {np.max(lens)}.")
    print(f"The minimum input length: {np.min(lens)}.")
    print(f"The average input length: {np.mean(lens)}.")
    
    print("GOOD BYE.")
        