from tqdm import tqdm

import os
import json
import pickle


def process_data(args, finetune_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/sim"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."

    finetune_dir = f"{finetune_dir}/sim"
    if not os.path.isdir(finetune_dir):
        os.makedirs(finetune_dir)

    action_class_dict = {}
    
    data_types = ['train', 'dev', 'test']
    num_train_dialogues = 0
    num_train_utters = 0
    num_valid_dialogues = 0
    num_valid_utters = 0
    num_test_dialogues = 0
    num_test_utters = 0
    for data_type in data_types:
        print(f"Processing {data_type} set...")
        utter_dialogues, action_dialogues, action_class_dict = load_dialogues(data_dir, data_type, action_class_dict, args)
        
        if data_type == 'train':
            num_train_dialogues = len(utter_dialogues)
            num_train_utters = count_utters(utter_dialogues)
            prefix = data_type
        elif data_type == 'dev':
            num_valid_dialogues = len(utter_dialogues)
            num_valid_utters = count_utters(utter_dialogues)
            prefix = args.valid_prefix
        elif data_type == 'test':
            num_test_dialogues = len(utter_dialogues)
            num_test_utters = count_utters(utter_dialogues)
            prefix = data_type

        save_file(finetune_dir, prefix, utter_dialogues, action_dialogues)
        
    print("Saving action class dictionary...")
    with open(f"{finetune_dir}/action_{args.class_dict_name}.json", 'w') as f:
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
    
    domains = ['Movie', 'Restaurant']
    for domain in domains:
        with open(f"{data_dir}/sim-{domain[0]}/{data_type}.json") as f:
            data = json.load(f)

        for dialogue in tqdm(data):
            turns = dialogue['turns']

            utter_dialogue = []
            action_dialogue = []
            for t, turn in enumerate(turns):
                if 'system_acts' in turn:
                    sys_utter = turn['system_utterance']['text']
                    sys_slots = turn['system_utterance']['slots']
                    sys_tokens = turn['system_utterance']['tokens']
                    sys_acts = turn['system_acts']
                    
                    utter_dialogue.append(f"sys:{sys_utter}")
                    
                    action_list, action_class_dict = find_actions(domain, sys_acts, action_class_dict, speaker='sys')
                    action_dialogue.append(action_list)
                
                usr_utter = turn['user_utterance']['text']
                usr_slots = turn['user_utterance']['slots']
                usr_tokens = turn['user_utterance']['tokens']
                usr_acts = turn['user_acts']
                
                utter_dialogue.append(f"usr:{usr_utter}")
    
                action_list, action_class_dict = find_actions(domain, usr_acts, action_class_dict, speaker='usr')
                action_dialogue.append(action_list)
            
            assert len(utter_dialogue) == len(action_dialogue)
            
            utter_dialogues.append(utter_dialogue)
            action_dialogues.append(action_dialogue)
            
    assert len(utter_dialogues) == len(action_dialogues)
            
    return utter_dialogues, action_dialogues, action_class_dict


def find_actions(domain, acts, action_class_dict, speaker):
    action_list = []
    
    for obj in acts:
        action = obj['type']
        
        if action not in action_class_dict and speaker == 'sys':
            action_class_dict[action] = len(action_class_dict)
            
        action_list.append((domain, action))
        
    return list(set(action_list)), action_class_dict


def save_file(finetune_dir, prefix, utter_dialogues, action_dialogues):
    with open(f"{finetune_dir}/{prefix}_utters.pickle", 'wb') as f:
        pickle.dump(utter_dialogues, f)
        
    with open(f"{finetune_dir}/{prefix}_actions.pickle", 'wb') as f:
        pickle.dump(action_dialogues, f)
        

def count_utters(dialogues):
    count = 0
    for dialogue in dialogues:
        count += len(dialogue)
        
    return count
        
