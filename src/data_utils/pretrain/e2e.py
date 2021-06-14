from tqdm import tqdm
from glob import glob

import os
import pickle
import json
import csv
import random
random.seed(777)


def process_data(args, pretrain_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/msr-e2e"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."
    
    pretrain_dir = f"{args.data_dir}/{args.pretrain_dir}/e2e"
    if not os.path.isdir(pretrain_dir):
        os.makedirs(pretrain_dir)

    action_map = {
        "Deny": "NEGATIVE",
        "Not_Sure": "GENERAL",
        "closing": "GENERAL",
        "confirm_answer": "POSITIVE",
        "confirm_question": "ASK",
        "greeting": "GENERAL",
        "inform": "INFORM",
        "multiple_choice": "OFFER",
        "request": "ASK",
        "thanks": "GENERAL",
        "welcome": "GENERAL",
    }
    
    print("Saving action map...")
    with open(f"{pretrain_dir}/{args.action_map_name}.json", 'w') as f:
        json.dump(action_map, f)
       
    domain_list = glob(f"{data_dir}/*")
    domain_list = [domain_file for domain_file in domain_list if domain_file.endswith('.tsv')]    
    
    print("Processing data...")
    train_utters, train_actions = [], []
    valid_utters, valid_actions = [], []
    for domain_file in domain_list:
        print(f"Processing {domain_file}...")
        utters, actions = [], []
        with open(domain_file, 'r') as f:
            data = list(csv.reader(f, delimiter='\t'))
        
        utter_hists, action_hists = [], []
        is_skip = False
        for l, line in enumerate(tqdm(data)):
            if l > 0:
                sess_id = line[0]
                text = line[4]
                action_tags = line[5:]
                
                if text == "":
                    is_skip = True
                    
                if not is_skip:
                    utter_hists.append(text)
                    action_set = set()
                    for action_tag in action_tags:
                        if len(action_tag) > 0:
                            idx = action_tag.find("(")
                            action_set.add(action_map[action_tag[:idx]])
                            action_hists.append(list(action_set))
                        
                if l+1 < len(data):
                    next_sess_id = data[l+1][0]
                    if next_sess_id != sess_id:
                        utters.append(utter_hists)
                        actions.append(action_hists)
                        utter_hists, actions_hists = [], []
                        is_skip = False
                else:
                    utters.append(utter_hists)
                    actions.append(action_hists)
                    
        domain_train_utters, domain_train_actions, domain_valid_utters, domain_valid_actions = split_data(utters, actions, args.train_frac)
        train_utters += domain_train_utters
        train_actions += domain_train_actions
        valid_utters += domain_valid_utters
        valid_actions += domain_valid_actions
    
    print("Saving each file...")
    save_files(pretrain_dir, train_utters, train_actions, args.train_prefix)
    save_files(pretrain_dir, valid_utters, valid_actions, args.valid_prefix)

    train_num_utters = count_utters(train_utters)
    valid_num_utters = count_utters(valid_utters)
    
    print("<Data Anaysis>")
    print(f"# of train dialogues: {len(train_utters)}")
    print(f"# of valid dialogues: {len(valid_utters)}")
    print(f"# of train utterances: {train_num_utters}")
    print(f"# of valid utterances: {valid_num_utters}")
    
    print("Done.")


def split_data(utters, actions, train_frac):
    pairs = list(zip(utters, actions))
    random.shuffle(pairs)
    utters, actions = zip(*pairs)
    train_utters, train_actions = utters[:int(len(utters) * train_frac)], actions[:int(len(actions) * train_frac)]
    valid_utters, valid_actions = utters[int(len(utters) * train_frac):], actions[int(len(actions) * train_frac):]
    
    return train_utters, train_actions, valid_utters, valid_actions


def save_files(pretrain_dir, utters, actions, prefix):
    with open(f"{pretrain_dir}/{prefix}_utters.pickle", 'wb') as f:
        pickle.dump(utters, f)
        
    with open(f"{pretrain_dir}/{prefix}_actions.pickle", 'wb') as f:
        pickle.dump(actions, f)


def count_utters(dialogues):
    count = 0
    for dialogue in dialogues:
        count += len(dialogue)
        
    return count
    
