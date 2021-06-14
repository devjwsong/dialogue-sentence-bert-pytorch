from tqdm import tqdm

import os
import json
import pickle


def process_data(args, pretrain_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/dstc8-schema-guided-dialogue-master"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."

    pretrain_dir = f"{args.data_dir}/{args.pretrain_dir}/schema"
    if not os.path.isdir(pretrain_dir):
        os.makedirs(pretrain_dir)
    
    action_map = {
        "INFORM": "INFORM",
        "REQUEST": "ASK",
        "CONFIRM": "ASK",
        "OFFER": "OFFER",
        "NOTIFY_SUCCESS": "POSITIVE",
        "NOTIFY_FAILURE": "NEGATIVE",
        "INFORM_COUNT": "INFORM",
        "OFFER_INTENT": "OFFER",
        "REQ_MORE": "ASK",
        "GOODBYE": "GENERAL",
        "INFORM_INTENT": "INFORM",
        "NEGATE_INTENT": "NEGATIVE",
        "AFFIRM_INTENT": "POSITIVE",
        "AFFIRM": "POSITIVE",
        "NEGATE": "NEGATIVE",
        "SELECT": "INFORM",
        "REQUEST_ALTS": "ASK",
        "THANK_YOU": "GENERAL"
    }
    
    print("Saving action map...")
    with open(f"{pretrain_dir}/{args.action_map_name}.json", 'w') as f:
        json.dump(action_map, f)
        
    type_list = ['train', 'dev', 'test']
    train_utters, train_actions = [], []
    valid_utters, valid_actions = [], []
    for type in type_list:
        print(f"Processing {type} data...")
        utters, actions = [], []
        dialogue_list = [dialogue for dialogue in os.listdir(f"{data_dir}/{type}") if dialogue.startswith('dialogues')]
        for dialogue in tqdm(dialogue_list):
            with open(f"{data_dir}/{type}/{dialogue}", 'r') as f:
                objs = json.load(f)
            for obj in objs:
                turns = obj['turns']
                
                utter_hists, action_hists = [], []
                for t, turn in enumerate(turns):
                    speaker = turn['speaker']
                    utter = turn['utterance']
        
                    if len(utter) == 0:
                        break
                    
                    utter_hists.append(utter)
                    action_tags = turn['frames'][0]['actions']
                    action_tags = list(set([action_map[action_tag['act']] for action_tag in action_tags]))
                    action_hists.append(action_tags)
                    
                utters.append(utter_hists)
                actions.append(action_hists)
                
        if type == 'train' or type == 'dev':
            train_utters += utters
            train_actions += actions
        else:
            valid_utters += utters
            valid_actions += actions
    
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
    
