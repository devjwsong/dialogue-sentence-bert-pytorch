from tqdm import tqdm

import os
import json
import pickle
import random
random.seed(0)


def process_data(args):
    data_dir = f"{args.data_dir}/{args.raw_dir}/MultiWOZ2_3"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."

    save_dir = f"{args.finetune_dir}/multiwoz"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    action_class_dict = {}

    domain2idxs = {}
    
    print("Loading dialogue data...")
    with open(f"{data_dir}/data.json", 'r') as f:
        data = json.load(f)
   
    print("Parsing dialogues...")
    utter_dialogues = []
    action_dialogues = []
    for dialogue_id, dialogue in tqdm(data.items()):
        domains = [k for k, v in dialogue['goal'].items() if k != 'topic' and k != 'message' and len(v) > 0]
        
        utter_dialogue = []
        action_dialogue = []
        
        for t, turn in enumerate(dialogue['log']):
            dialog_act = turn['dialog_act']
            metadata = turn['metadata']
            span_info = turn['span_info']
            
            if len(metadata) == 0:  # User
                speaker = 'usr'
            else:  # System
                speaker = 'sys'
                
            action_list, action_class_dict = find_actions(dialog_act, action_class_dict, speaker)
            
            utter = turn['text'].replace('\n', '')
            
            utter_dialogue.append(f"{speaker}:{utter}")
            action_dialogue.append(action_list)

        assert len(utter_dialogue) == len(action_dialogue)
        
        utter_dialogues.append(utter_dialogue)
        action_dialogues.append(action_dialogue)
        
        idx = len(utter_dialogues)-1
        domain = random.sample(domains, 1)[0]
        if domain not in domain2idxs:
            domain2idxs[domain] = []
        domain2idxs[domain].append(idx)
    
    assert len(utter_dialogues) == len(action_dialogues)
    
    num_total_dialogs = 0
    for k, v in domain2idxs.items():
        num_total_dialogs += len(v)
        
        print(f"{k}: {len(v)}")
        
    assert num_total_dialogs == len(utter_dialogues)
    
    train_utter_dialogs = []
    valid_utter_dialogs = []
    test_utter_dialogs = []
    train_action_dialogs = []
    valid_action_dialogs = []
    test_action_dialogs = []
    
    print("Splitting data...")
    for domain, idxs in domain2idxs.items():
        train_idxs, valid_idxs, test_idxs = split_data(idxs, args.train_frac, args.valid_frac)
        
        train_utter_dialogs += [utter_dialogues[idx] for idx in train_idxs]
        valid_utter_dialogs += [utter_dialogues[idx] for idx in valid_idxs]
        test_utter_dialogs += [utter_dialogues[idx] for idx in test_idxs]
        
        train_action_dialogs += [action_dialogues[idx] for idx in train_idxs]
        valid_action_dialogs += [action_dialogues[idx] for idx in valid_idxs]
        test_action_dialogs += [action_dialogues[idx] for idx in test_idxs]
    
    print("Now saving data...")
    save_file(save_dir, args.train_prefix, train_utter_dialogs, train_action_dialogs)
    save_file(save_dir, args.valid_prefix, valid_utter_dialogs, valid_action_dialogs)
    save_file(save_dir, args.test_prefix, test_utter_dialogs, test_action_dialogs)
        
    print("Saving action class dictionary...")
    with open(f"{save_dir}/action_{args.class_dict_name}.json", 'w') as f:
        json.dump(action_class_dict, f)
    
    num_train_utters = count_utters(train_utter_dialogs)
    num_valid_utters = count_utters(valid_utter_dialogs)
    num_test_utters = count_utters(test_utter_dialogs)
    
    print("<Data Anaysis>")
    print("Task: Action Prediction")
    print(f"# of train dialogues: {len(train_utter_dialogs)}")
    print(f"# of train utterances: {num_train_utters}")
    print(f"# of valid dialogues: {len(valid_utter_dialogs)}")
    print(f"# of valid utterances: {num_valid_utters}")
    print(f"# of test dialogues: {len(test_utter_dialogs)}")
    print(f"# of test utterances: {num_test_utters}")
    print(f"# of classes: {len(action_class_dict)}")
    
    print("Done.")


def find_actions(dialog_act, action_class_dict, speaker):
    action_list = []
    for act, _ in dialog_act.items():
        domain = act.split('-')[0]
        action = act.split('-')[1]
        if action not in action_class_dict and speaker == 'sys':
            action_class_dict[action] = len(action_class_dict)
            
        action_list.append((domain, action))
    
    return list(set(action_list)), action_class_dict


def split_data(idxs, train_frac, valid_frac):
    random.seed(111)
    random.shuffle(idxs)
    train_idxs = idxs[:int(len(idxs) * train_frac)]
    remained_idxs = idxs[int(len(idxs) * train_frac):]
    
    f = valid_frac / (1.0-train_frac)
    valid_idxs = remained_idxs[:int(len(remained_idxs) * f)]
    test_idxs = remained_idxs[int(len(remained_idxs) * f):]
    
    return train_idxs, valid_idxs, test_idxs


def save_file(save_dir, prefix, utter_dialogues, action_dialogues):
    with open(f"{save_dir}/{prefix}_utters.pickle", 'wb') as f:
        pickle.dump(utter_dialogues, f)
        
    with open(f"{save_dir}/{prefix}_actions.pickle", 'wb') as f:
        pickle.dump(action_dialogues, f)
        

def count_utters(dialogues):
    count = 0
    for dialogue in dialogues:
        count += len(dialogue)
        
    return count
    
