from tqdm import tqdm

import os
import json
import pickle


def process_data(args, processed_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/sim"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."

    processed_dir = f"{processed_dir}/sim"
    if not os.path.isdir(processed_dir):
        os.makedirs(processed_dir)

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

        save_file(processed_dir, prefix, utter_dialogues, entity_dialogues, action_dialogues)

    print("Saving entity class dictionary...")
    with open(f"{processed_dir}/entity_{args.class_dict_name}.json", 'w') as f:
        json.dump(entity_class_dict, f)
        
    print("Saving action class dictionary...")
    with open(f"{processed_dir}/action_{args.class_dict_name}.json", 'w') as f:
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
                    sys_slots = turn['system_utterance']['slots']
                    sys_tokens = turn['system_utterance']['tokens']
                    sys_acts = turn['system_acts']
                    
                    utter_dialogue.append(f"sys:{sys_utter}")
                    
                    entity_list, entity_class_dict = find_entities(domain, sys_slots, sys_tokens, entity_class_dict, speaker='sys')
                    action_list, action_class_dict = find_actions(domain, sys_acts, action_class_dict, speaker='sys')
            
                    entity_dialogue.append(entity_list)
                    action_dialogue.append(action_list)
                
                usr_utter = turn['user_utterance']['text']
                usr_slots = turn['user_utterance']['slots']
                usr_tokens = turn['user_utterance']['tokens']
                usr_acts = turn['user_acts']
                
                utter_dialogue.append(f"usr:{usr_utter}")
                
                entity_list, entity_class_dict = find_entities(domain, usr_slots, usr_tokens, entity_class_dict, speaker='usr')
                action_list, action_class_dict = find_actions(domain, usr_acts, action_class_dict, speaker='usr')
                
                entity_dialogue.append(entity_list)
                action_dialogue.append(action_list)
            
            assert len(utter_dialogue) == len(entity_dialogue)
            assert len(utter_dialogue) == len(action_dialogue)
            
            utter_dialogues.append(utter_dialogue)
            entity_dialogues.append(entity_dialogue)
            action_dialogues.append(action_dialogue)
            
    assert len(utter_dialogues) == len(entity_dialogues)
    assert len(utter_dialogues) == len(action_dialogues)
            
    return utter_dialogues, entity_dialogues, action_dialogues, entity_class_dict, action_class_dict


def find_entities(domain, slots, tokens, entity_class_dict, speaker):
    entity_list = []
    
    for slot in slots:
        entity_type = slot['slot']
        if f"B-{domain}-{entity_type}" not in entity_class_dict and speaker=='usr':
            entity_class_dict[f"B-{domain}-{entity_type}"] = len(entity_class_dict)
            entity_class_dict[f"I-{domain}-{entity_type}"] = len(entity_class_dict)
        
        start = slot['start']
        exclusive_end = slot['exclusive_end']
        entity_list.append((f"{domain}-{entity_type}", ' '.join(tokens[start:exclusive_end]), start, exclusive_end))
    
    return entity_list, entity_class_dict


def find_actions(domain, acts, action_class_dict, speaker):
    action_list = []
    
    for obj in acts:
        action = obj['type']
        
        if action not in action_class_dict and speaker == 'sys':
            action_class_dict[action] = len(action_class_dict)
            
        action_list.append((domain, action))
        
    return list(set(action_list)), action_class_dict


def save_file(processed_dir, prefix, utter_dialogues, entity_dialogues, action_dialogues):
    with open(f"{processed_dir}/{prefix}_utters.pickle", 'wb') as f:
        pickle.dump(utter_dialogues, f)

    with open(f"{processed_dir}/{prefix}_entities.pickle", 'wb') as f:
        pickle.dump(entity_dialogues, f)
        
    with open(f"{processed_dir}/{prefix}_actions.pickle", 'wb') as f:
        pickle.dump(action_dialogues, f)
        

def count_utters(dialogues):
    count = 0
    for dialogue in dialogues:
        count += len(dialogue)
        
    return count
        
