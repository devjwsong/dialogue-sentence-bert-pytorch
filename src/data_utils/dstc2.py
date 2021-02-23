from tqdm import tqdm

import os
import json
import pickle


def process_data(args, finetune_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/dstc2"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."
        
    # For action prediction
    action_dir = f"{finetune_dir}/{args.action_dir}/dstc2"
    if not os.path.isdir(action_dir):
        os.makedirs(action_dir)
    
    action_class_dict = {}
    
    data_list = ['traindev','test']
    num_train_dialogues = 0
    num_valid_dialogues = 0
    num_test_dialogues = 0
    num_train_utters = 0
    num_train_utters = 0
    num_train_utters = 0
    for data_type in data_list:
        
        print(f"Processing {data_type} set...")
        utter_dialogues, action_dialogues, action_class_dict = load_dialogues(data_dir, data_type, action_class_dict, args)
        
        if data_type == 'traindev':
            train_utter_dialogues, valid_utter_dialogues = split_data(utter_dialogues)
            train_action_dialogues, valid_action_dialogues = split_data(action_dialogues)
    
            num_train_dialogues = len(train_utter_dialogues)
            num_train_utters = count_utters(train_utter_dialogues)
            num_valid_dialogues = len(valid_utter_dialogues)
            num_valid_utters = count_utters(valid_utter_dialogues)

            save_files("Action Prediction", action_dir, args.train_prefix, train_utter_dialogues, train_action_dialogues, args)
            save_files("Action Prediction", action_dir, args.valid_prefix, valid_utter_dialogues, valid_action_dialogues, args)
        else:
            num_test_dialogues = len(utter_dialogues)
            num_test_utters = count_utters(utter_dialogues)
            save_files("Action Prediction", action_dir, args.test_prefix, utter_dialogues, action_dialogues, args)
        
    print("Saving action class dictionary...")
    with open(f"{action_dir}/{args.class_dict_name}.json", 'w') as f:
        json.dump(action_class_dict, f)
    
    print("<Data Anaysis>")
    print("Task: Action Prediction")
    print(f"# of train dialogues: {num_train_dialogues}")
    print(f"# of train utterances: {num_train_utters}")
    print(f"# of valid dialogues: {num_valid_dialogues}")
    print(f"# of valid utterances: {num_valid_utters}")
    print(f"# of test dialogues: {num_test_dialogues}")
    print(f"# of test utterances: {num_test_utters}")
    print(f"# of classes: {len(action_class_dict)}")
    
    print("Done.")
    

def load_dialogues(data_dir, data_type, action_class_dict, args):
    utter_dialogues = []
    action_dialogues = []
    
    dir_list = os.listdir(f"{data_dir}/{data_type}/data")
    for dir_name in dir_list:
        dialogue_list = os.listdir(f"{data_dir}/{data_type}/data/{dir_name}")
        for dialogue in tqdm(dialogue_list):
            utter_dialogue = []
            action_dialogue = []
            
            with open(f"{data_dir}/{data_type}/data/{dir_name}/{dialogue}/log.json") as f:
                log = json.load(f)
            with open(f"{data_dir}/{data_type}/data/{dir_name}/{dialogue}/label.json") as f:
                label = json.load(f)
                
            log_turns = log['turns']
            label_turns = label['turns']
            
            for t, turn in enumerate(log_turns):
                sys_utter = turn['output']['transcript']
                usr_utter = label_turns[t]['transcription']
                
                utter_dialogue.append(f"speaker2:{sys_utter}")
                utter_dialogue.append(f"speaker1:{usr_utter}")
                
                dialog_acts = turn['output']['dialog-acts'] # System side
                
                action_list, action_class_dict = find_actions(dialog_acts, action_class_dict)
                
                action_dialogue += [[], []]
                if t>0:
                    action_dialogue[2*t-1] = action_list
            
            assert len(utter_dialogue) == len(action_dialogue)
            
            utter_dialogues.append(utter_dialogue)
            action_dialogues.append(action_dialogue)
    
    assert len(utter_dialogues) == len(action_dialogues)
            
    return utter_dialogues, action_dialogues, action_class_dict


def find_actions(dialog_acts, action_class_dict):
    action_list = []
    
    for obj in dialog_acts:
        action = obj['act']
        
        if action not in action_class_dict:
            action_class_dict[action] = len(action_class_dict)
            
        action_list.append(action)
        
    return list(set(action_list)), action_class_dict


def split_data(dialogues, train_frac=0.75):    
    train_dialogues = dialogues[:int(len(dialogues) * train_frac)]
    valid_dialogues = dialogues[int(len(dialogues) * train_frac):]
    
    return train_dialogues, valid_dialogues


def save_files(task_name, save_dir, prefix, utter_dialogues, label_dialogues, args):
    print(f"Saving {prefix} data for {task_name} as pickle...")
    with open(f"{save_dir}/{prefix}_{args.utter_name}.pickle", 'wb') as f:
        pickle.dump(utter_dialogues, f)
        
    with open(f"{save_dir}/{prefix}_{args.label_name}.pickle", 'wb') as f:
        pickle.dump(label_dialogues, f)
        
        
def count_utters(dialogues):
    count = 0
    for dialogue in dialogues:
        count += len(dialogue)
        
    return count
        