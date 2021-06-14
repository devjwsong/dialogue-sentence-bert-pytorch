from tqdm import tqdm

import os
import json
import pickle
import random
random.seed(777)


def process_data(args, pretrain_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/frames"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."

    pretrain_dir = f"{args.data_dir}/{args.pretrain_dir}/frames"
    if not os.path.isdir(pretrain_dir):
        os.makedirs(pretrain_dir)
    
    action_map = {
        "inform": "INFORM",
        "offer": "OFFER",
        "request": "ASK",
        "switch_frame": "GENERAL",
        "suggest": "OFFER",
        "no_result": "INFORM",
        "thankyou": "GENERAL",
        "sorry": "GENERAL",
        "greeting": "GENERAL",
        "affirm": "POSITIVE",
        "negate": "NEGATIVE",
        "confirm": "ASK",
        "moreinfo": "ASK",
        "goodbye": "GENERAL",
        "request_alts": "ASK",
        "request_compare": "ASK",
        "hearmore": "ASK",
        "you_are_welcome": "GENERAL",
        "canthelp": "GENERAL",
        "reject": "GENERAL"
    }
    
    print("Saving action map...")
    with open(f"{pretrain_dir}/{args.action_map_name}.json", 'w') as f:
        json.dump(action_map, f)
    
    with open(f"{data_dir}/frames.json", 'r') as f:
        data = json.load(f)
    
    print("Processing data...")
    pairs = []
    utters, actions = [], []
    for dialogue in tqdm(data):
        turns = dialogue['turns']
        utter_hists, action_hists = [], []
        for t, turn in enumerate(turns):
            text = turn['text']
            utter_hists.append(text)
            
            action_tags = []
            if 'acts' in turn['labels']:
                action_tags += turn['labels']['acts']
            if 'acts_without_refs' in turn['labels']:
                action_tags += turn['labels']['acts_without_refs']

            action_tags = list(set([action_map[action_tag['name']] for action_tag in action_tags]))
            action_hists.append(action_tags)
            
        utters.append(utter_hists)
        actions.append(action_hists)
            
    print("Shuffing & spliting data...")
    train_utters, train_actions, valid_utters, valid_actions = split_data(utters, actions, args.train_frac)
    
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
    
