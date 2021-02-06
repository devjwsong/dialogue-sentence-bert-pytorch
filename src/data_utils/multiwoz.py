from tqdm import tqdm

import os
import json
import pickle


def process_data(args, finetune_dir):
    data_dir = f"{args.data_dir}/{args.raw_dir}/MultiWOZ2_3"
    assert os.path.isdir(data_dir), "Please check the raw data directory path."
    
#     # For dialogue state tracking
#     state_dir = f"{finetune_dir}/{args.state_dir}/multiwoz"
#     if not os.path.isdir(state_dir):
#         os.makedirs(state_dir)

    # For entity recogntion
    entity_dir = f"{finetune_dir}/{args.entity_dir}/multiwoz"
    if not os.path.isdir(entity_dir):
        os.makedirs(entity_dir)
        
    # For action prediction
    action_dir = f"{finetune_dir}/{args.action_dir}/multiwoz"
    if not os.path.isdir(action_dir):
        os.makedirs(action_dir)
    
#     state_class_dict = {}
    entity_class_dict = {'O': 0}
    action_class_dict = {}
    
#     print("Loading & processing ontologies...")
#     with open(f"{data_dir}/ontology.json", 'r') as f:
#         onts = json.load(f)
    
#     for pair, value_list in tqdm(onts.items()):
#         if pair not in state_class_dict:
#             state_class_dict[pair] = [len(state_class_dict), {args.none_value: 0}]
        
#         for i, value in enumerate(value_list):
#             if value not in state_class_dict[pair][1]:
#                 state_class_dict[pair][1][value] = len(state_class_dict[pair][1])
    
    print("Loading dialogue data...")
    with open(f"{data_dir}/data.json", 'r') as f:
        data = json.load(f)
   
    print("Processing dialogues...")
    utter_dialogues = []
#     state_dialogues = []
    entity_dialogues = []
    action_dialogues = []
    for dialogue_id, dialogue in tqdm(data.items()):
        utter_dialogue = []
#         state_dialogue = []
        entity_dialogue = []
        action_dialogue = []
        
        for t, turn in enumerate(dialogue['log']):
            dialog_act = turn['dialog_act']
            metadata = turn['metadata']
            span_info = turn['span_info']
            if len(metadata) == 0:  # User
                speaker = 'speaker1'
#                 state_ids = []
                entity_list, entity_class_dict = find_entities(span_info, entity_class_dict)
                action_list = []
            else:  # System
                speaker = 'speaker2'
#                 state_ids = find_states(metadata, state_class_dict, args.none_value)
                entity_list = []
                action_list, action_class_dict = find_actions(dialog_act, action_class_dict)
                
                if t>0:
                    action_dialogue[t-1] = action_list
                    action_list = []
            
            utter = turn['text'].replace('\n', '')
            
            utter_dialogue.append(f"{speaker}:{utter}")
#             state_dialogue.append(state_list)
            entity_dialogue.append(entity_list)
            action_dialogue.append(action_list)
        
#         assert len(utter_dialogue) == len(state_dialogue)
        assert len(utter_dialogue) == len(entity_dialogue)
        assert len(utter_dialogue) == len(action_dialogue)
        
        utter_dialogues.append(utter_dialogue)
#         state_dialogues.append(state_dialogue)
        entity_dialogues.append(entity_dialogue)
        action_dialogues.append(action_dialogue)
    
#     assert len(utter_dialogues) == len(state_dialogues)
    assert len(utter_dialogues) == len(entity_dialogues)
    assert len(utter_dialogues) == len(action_dialogues)
    
    print("Splitting data...")
    train_utter_dialogues, valid_utter_dialogues, test_utter_dialogues = \
        split_data(utter_dialogues, args.train_frac, args.valid_frac)
#     train_state_dialogues, valid_state_dialogues, test_state_dialogues = \
#         split_data(state_dialogues, args.train_frac, args.valid_frac)
    train_entity_dialogues, valid_entity_dialogues, test_entity_dialogues = \
        split_data(entity_dialogues, args.train_frac, args.valid_frac)
    train_action_dialogues, valid_action_dialogues, test_action_dialogues = \
        split_data(action_dialogues, args.train_frac, args.valid_frac)
    
    print("Now saving data...")
#     save_files("Dialogue State Tracking", state_dir, args.train_prefix, train_utter_dialogues, train_state_dialogues, args)
    save_files("Entity Recognition", entity_dir, args.train_prefix, train_utter_dialogues, train_entity_dialogues, args)
    save_files("Action Prediction", action_dir, args.train_prefix, train_utter_dialogues, train_action_dialogues, args)
    
#     save_files("Dialogue State Tracking", state_dir, args.valid_prefix, valid_utter_dialogues, valid_state_dialogues, args)
    save_files("Entity Recognition", entity_dir, args.valid_prefix, valid_utter_dialogues, valid_entity_dialogues, args)
    save_files("Action Prediction", action_dir, args.valid_prefix, valid_utter_dialogues, valid_action_dialogues, args)
    
#     save_files("Dialogue State Tracking", state_dir, args.test_prefix, test_utter_dialogues, test_state_dialogues, args)
    save_files("Entity Recognition", entity_dir, args.test_prefix, test_utter_dialogues, test_entity_dialogues, args)
    save_files("Action Prediction", action_dir, args.test_prefix, test_utter_dialogues, test_action_dialogues, args)
    
#     print("Saving state class dictionary...")
#     with open(f"{state_dir}/{args.class_dict_name}.json", 'w') as f:
#         json.dump(state_class_dict, f)

    print("Saving entity class dictionary...")
    with open(f"{entity_dir}/{args.class_dict_name}.json", 'w') as f:
        json.dump(entity_class_dict, f)
        
    print("Saving action class dictionary...")
    with open(f"{action_dir}/{args.class_dict_name}.json", 'w') as f:
        json.dump(action_class_dict, f)
    
    num_train_utters = count_utters(train_utter_dialogues)
    num_valid_utters = count_utters(valid_utter_dialogues)
    num_test_utters = count_utters(test_utter_dialogues)
    
    print("<Data Anaysis>")
    
    print("Task: Entity Recognition")
    print(f"# of train dialogues: {len(train_utter_dialogues)}")
    print(f"# of train utterances: {num_train_utters}")
    print(f"# of valid dialogues: {len(valid_utter_dialogues)}")
    print(f"# of valid utterances: {num_valid_utters}")
    print(f"# of test dialogues: {len(test_utter_dialogues)}")
    print(f"# of test utterances: {num_test_utters}")
    print(f"# of classes: {len(entity_class_dict)}")
    
    print("Task: Action Prediction")
    print(f"# of train dialogues: {len(train_utter_dialogues)}")
    print(f"# of train utterances: {num_train_utters}")
    print(f"# of valid dialogues: {len(valid_utter_dialogues)}")
    print(f"# of valid utterances: {num_valid_utters}")
    print(f"# of test dialogues: {len(test_utter_dialogues)}")
    print(f"# of test utterances: {num_test_utters}")
    print(f"# of classes: {len(action_class_dict)}")
    
    print("Done.")


def find_entities(span_info, entity_class_dict):
    entity_list = []
    for span in span_info:
        domain = span[0].split('-')[0]
        slot_type = span[1]
        entity_value = span[2]
        start = span[3]
        end = span[4]
        
        entity_type = f"{domain}-{slot_type}"
        if f"B-{entity_type}" not in entity_class_dict:
            entity_class_dict[f"B-{entity_type}"] = len(entity_class_dict)
            entity_class_dict[f"I-{entity_type}"] = len(entity_class_dict)
            
        entity_list.append((entity_type, entity_value, start, end))
        
    return entity_list, entity_class_dict

    
# def find_states(metadata, state_class_dict, none_value):
#     state_ids = []
#     for domain, info in metadata.items():
#         main = info['book']
#         sub = info['semi']
        
#         main.update(sub)
        
#         for slot_type, slot_value in main.items():
#             if slot_type != 'booked':
#                 if f"{domain}-book {slot_type}" in state_class_dict:
#                     pair_name = f"{domain}-book {slot_type}"
#                 elif f"{domain}-{slot_type}" in state_class_dict:
#                     pair_name = f"{domain}-{slot_type}"
#                 else:
#                     continue
                    
#                 if slot_value not in state_class_dict[pair_name][1]:
#                     state_ids.append((state_class_dict[pair_name][0], state_class_dict[pair_name][1][none_value]))
#                 else:
#                     state_ids.append((state_class_dict[pair_name][0], state_class_dict[pair_name][1][slot_value]))
                    
#     return state_ids


def find_actions(dialog_act, action_class_dict):
    action_list = []
    for act, _ in dialog_act.items():
        if act not in action_class_dict:
            action_class_dict[act] = len(action_class_dict)
            
        action_list.append(act)
    
    return list(set(action_list)), action_class_dict


def split_data(dialogues, train_frac, valid_frac):
    train_dialogues = dialogues[:int(len(dialogues) * train_frac)]
    remained_dialogues = dialogues[int(len(dialogues) * train_frac):]
    
    f = valid_frac / (1.0-train_frac)
    valid_dialogues = remained_dialogues[:int(len(remained_dialogues) * f)]
    test_dialogues = remained_dialogues[int(len(remained_dialogues) * f):]
    
    return train_dialogues, valid_dialogues, test_dialogues


def save_files(task_name, save_dir, prefix, utter_dialogues, label_dialogues, args):
    print(f"Saving {prefix} data for {task_name} as pickle files...")
    with open(f"{save_dir}/{prefix}_{args.utter_name}.pickle", 'wb') as f:
        pickle.dump(utter_dialogues, f)
        
    with open(f"{save_dir}/{prefix}_{args.label_name}.pickle", 'wb') as f:
        pickle.dump(label_dialogues, f)
        

def count_utters(dialogues):
    count = 0
    for dialogue in dialogues:
        count += len(dialogue)
        
    return count
    