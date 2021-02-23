from tqdm import tqdm

import os
import json
import pickle


def process_data(args, processed_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/sim"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."

    # For entity recognition
    entity_dir = f"{processed_dir}/{args.entity_dir}/sim"
    if not os.path.isdir(entity_dir):
        os.makedirs(entity_dir)
        
    # For action prediction
    action_dir = f"{processed_dir}/{args.action_dir}/sim"
    if not os.path.isdir(action_dir):
        os.makedirs(action_dir)

    entity_class_dict = {'O': 0}
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
        utter_dialogues, entity_dialogues, action_dialogues, entity_class_dict, action_class_dict =\
            load_dialogues(data_dir, data_type, entity_class_dict, action_class_dict, args)
        
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

        save_files("Entity Recognition", entity_dir, prefix, utter_dialogues, entity_dialogues, args)
        save_files("Action Prediction", action_dir, prefix, utter_dialogues, action_dialogues, args)

    print("Saving entity class dictionary...")
    with open(f"{entity_dir}/{args.class_dict_name}.json", 'w') as f:
        json.dump(entity_class_dict, f)
        
    print("Saving action class dictionary...")
    with open(f"{action_dir}/{args.class_dict_name}.json", 'w') as f:
        json.dump(action_class_dict, f)
        
    print("<Data Anaysis>")
    
    print("Task: Entity Recognition")
    print(f"# of train dialogues: {num_train_dialogues}")
    print(f"# of train utterances: {num_train_utters}")
    print(f"# of valid dialogues: {num_valid_dialogues}")
    print(f"# of valid utterances: {num_valid_utters}")
    print(f"# of test dialogues: {num_test_dialogues}")
    print(f"# of test utterances: {num_test_utters}")
    print(f"# of classes: {len(entity_class_dict)}")
    
    print("Task: Action Prediction")
    print(f"# of train dialogues: {num_train_dialogues}")
    print(f"# of train utterances: {num_train_utters}")
    print(f"# of valid dialogues: {num_valid_dialogues}")
    print(f"# of valid utterances: {num_valid_utters}")
    print(f"# of test dialogues: {num_test_dialogues}")
    print(f"# of test utterances: {num_test_utters}")
    print(f"# of classes: {len(action_class_dict)}")
    
    print("Done.")

    
def load_dialogues(data_dir, data_type, entity_class_dict, action_class_dict, args):
    utter_dialogues = []
    entity_dialogues = []
    action_dialogues = []
    
    domains = ['Movie', 'Restaurant']
    for domain in domains:
        with open(f"{data_dir}/sim-{domain[0]}/{data_type}.json") as f:
            data = json.load(f)

        for dialogue in tqdm(data):
            turns = dialogue['turns']

            utter_dialogue = []
            entity_dialogue = []
            action_dialogue = []
            for t, turn in enumerate(turns):
                if 'system_acts' in turn:
                    sys_utter = turn['system_utterance']['text']
                    system_acts = turn['system_acts']
                    
                    utter_dialogue.append(f"speaker2:{sys_utter}")
                    
                    action_list, action_class_dict = find_actions(system_acts, action_class_dict)
            
                    entity_dialogue.append([])
                    action_dialogue.append([])
                    if t > 0:
                        action_dialogue[len(utter_dialogue)-2] = action_list
                
                usr_utter = turn['user_utterance']['text']
                slots = turn['user_utterance']['slots']
                tokens = turn['user_utterance']['tokens']
                
                utter_dialogue.append(f"speaker1:{usr_utter}")
                
                entity_list, entity_class_dict = find_entities(domain, slots, tokens, entity_class_dict)
                entity_dialogue.append(entity_list)
                action_dialogue.append([])
            
            assert len(utter_dialogue) == len(entity_dialogue)
            assert len(utter_dialogue) == len(action_dialogue)
            
            utter_dialogues.append(utter_dialogue)
            entity_dialogues.append(entity_dialogue)
            action_dialogues.append(action_dialogue)
            
    assert len(utter_dialogues) == len(entity_dialogues)
    assert len(utter_dialogues) == len(action_dialogues)
            
    return utter_dialogues, entity_dialogues, action_dialogues, entity_class_dict, action_class_dict


def find_entities(domain, slots, tokens, entity_class_dict):
    entity_list = []
    
    for slot in slots:
        entity_type = slot['slot']
        if f"B-{domain}-entity_type" not in entity_class_dict:
            entity_class_dict[f"B-{domain}-{entity_type}"] = len(entity_class_dict)
            entity_class_dict[f"I-{domain}-{entity_type}"] = len(entity_class_dict)
        
        start = slot['start']
        exclusive_end = slot['exclusive_end']
        entity_list.append((f"{domain}-{entity_type}", ' '.join(tokens[start:exclusive_end]), start, exclusive_end))
    
    return entity_list, entity_class_dict


def find_actions(system_acts, action_class_dict):
    action_list = []
    
    for obj in system_acts:
        action = obj['type']
        
        if action not in action_class_dict:
            action_class_dict[action] = len(action_class_dict)
            
        action_list.append(action)
        
    return list(set(action_list)), action_class_dict


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
        