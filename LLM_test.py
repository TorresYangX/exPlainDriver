import os
import json
import numpy as np
import pandas as pd
from pgm.PGM import PGM
from tqdm import tqdm
from pgm.config import *
from utils import Llama3_map_action, update_action, LLM_compare_action
from pgm.predicate_map import segment_to_vector

action_list = ["Keep", "Accelerate", "Decelerate", "Stop", "Reverse", "MakeLeftTurn", 
               "MakeRightTurn", "MakeUTurn", "Merge", "LeftPass", "RightPass", "Yield", 
               "ChangeToLeftLane", "ChangeToRightLane", "Park", "PullOver"]


def id2action(data, id):
    for item in data:
        if item["id"] == id:
            return item['action'], item
    return None, None



def LLM_acc(LLM_result_path, ground_truth_info, wrong_case_save_path):
    LLM_result = json.load(open(LLM_result_path))
    ground_truth_data = json.load(open(ground_truth_info))
    
    total = 0
    correct = 0
    special = 0
    wrong_case = []
    
    for item in tqdm(LLM_result):
        id = item['image_id']
        LLM_action_ = item['caption']
        ground_action_, _ = id2action(ground_truth_data, id)
        if ground_action_ is None:
            print(f"Ground Truth not found for id: {id}")
            continue
        
        total += 1
        LLM_action_ = LLM_action_.lower().strip()
        ground_action_ = ground_action_.lower().strip()
        if LLM_action_ == ground_action_:
            correct += 1
            continue
        
        LLM_action = Llama3_map_action(LLM_action_)
        ground_action = Llama3_map_action(ground_action_)
        
        LLM_action_set = update_action(action_list, LLM_action)
        ground_action_set = update_action(action_list, ground_action)
        
        if LLM_action_set == ground_action_set:
            correct += 1
            print(f"Same -- LLM: {LLM_action_set}, Ground Truth: {ground_action_set}")
        elif ground_action_set.issubset(LLM_action_set):
            correct += 1
            print(f"Include -- LLM: {LLM_action_set}, Ground Truth: {ground_action_set}")
        elif LLM_action_set.issubset(ground_action_set):
            special += 1
            print(f"Special -- LLM: {LLM_action_set}, Ground Truth: {ground_action_set}")
        else:
            wrong_case.append({
                'id': id,
                'LLM_action': LLM_action_,
                'ground_action': ground_action_,
                'LLM_action_set': list(LLM_action_set),
                'ground_action_set': list(ground_action_set)
            })
            print(f"Wrong -- LLM: {LLM_action_set}, \nGround Truth: {ground_action_set},\n id: {id},\n LLM: {LLM_action_},\n Ground Truth: {ground_action_}")
            
    acc = correct / total
    
    print(f"Total: {total}, Correct: {correct}, Special: {special}, Accuracy: {acc}")
    
    with open(wrong_case_save_path, 'w') as f:
        json.dump(wrong_case, f)
    
    return acc


def LLM_acc_direct(LLM_result_path, ground_truth_info, wrong_case_save_path):
    LLM_result = json.load(open(LLM_result_path))
    ground_truth_data = json.load(open(ground_truth_info))
    total = 0
    correct = 0
    wrong_case = []
    for item in tqdm(LLM_result):
        id = item['image_id']
        LLM_action_ = item['caption']
        ground_action_, _ = id2action(ground_truth_data, id)
        if ground_action_ is None:
            print(f"Ground Truth not found for id: {id}")
            continue
        LLM_ans = LLM_compare_action(LLM_action_, ground_action_).lower().strip()
        if 'yes' in LLM_ans:
            correct += 1
        else:
            wrong_case.append({
                'id': id,
                'LLM_action': LLM_action_,
                'ground_action': ground_action_
            })
        total += 1
        
    acc = correct / total
    print(f"Total: {total}, Correct: {correct}, Accuracy: {acc}")
    with open(wrong_case_save_path, 'w') as f:
        json.dump(wrong_case, f)
    return acc


def LLM_PGM_acc(weight_path, config, LLM_result_path, ground_truth_info, detection_result, wrong_case_save_path):
    pgm = PGM(weight_path=weight_path, config=config)
    LLM_result = json.load(open(LLM_result_path))
    ground_truth_data = json.load(open(ground_truth_info))
    yolo_detect_Test = json.load(open(detection_result))
    
    total = 0
    correct = 0
    wrong_case = []
    
    i = 0
    for item in tqdm(LLM_result):
        i += 1
        id = item['image_id']
        LLM_action_ = item['caption']
        ground_action_, video_info = id2action(ground_truth_data, id)
        
        if ground_action_ is None:
            continue
        
        total += 1
        LLM_action_ = LLM_action_.lower().strip()
        ground_action_ = ground_action_.lower().strip()
        
        LLM_action = Llama3_map_action(LLM_action_)
        LLM_action = update_action(action_list, LLM_action)
           
        yolo_class = None
        for item in yolo_detect_Test:
            if item['id'] == id:
                yolo_class = item['classes']
                print(f"YOLO: {yolo_class}, id: {id}")
                break
        instance = {
            "action": LLM_action,
            "classes": yolo_class
        }    
        
        instance_vector = np.array(segment_to_vector(instance))
        violate_rule = pgm.validate_instance(instance_vector)
        if violate_rule:
            nature_rule_v = config.mapping_natural_rule(violate_rule)
        
            #TODO: add violate rule to prompt and regenerate the action
            # LLM_action_ = model.generate(...)
            
        LLM_ans = LLM_compare_action(LLM_action_, ground_action_).lower().strip()
        if 'yes' in LLM_ans:
            correct += 1
        else:
            wrong_case.append({
                'id': id,
                'LLM_action': LLM_action_,
                'ground_action': ground_action_,
                'violate_rule': violate_rule
            })
        total += 1
    acc = correct / total
    print(f"Total: {total}, Correct: {correct}, Accuracy: {acc}")
    with open(wrong_case_save_path, 'w') as f:
        json.dump(wrong_case, f)
    return acc
            
            


if __name__ == "__main__":
    weight_save_path = 'optimal_weights_Interface.npy'
    yolo_detect_Test_path = 'process_data/test/test_detected_classes.json'
    save_redress_path = 'test_case/redress_case_0.0.json'
    
    LLM_result_path = 'Data/video_process/BDDX_Test_pred_action.json'
    ground_truth_path = 'process_data/test/map_ann_test.json'
    wrong_case_file = 'test_case/wrong_case.json'
    acc = LLM_acc_direct(LLM_result_path, ground_truth_path, wrong_case_file)
    print(acc)
    
    
    # LLM_PGM_acc(weight_save_path, LLM_result_path, ground_truth_path, yolo_detect_Test_path, save_redress_path)
    
    # acc, special_case = LLM_acc(LLM_result_path, ground_truth_path)
    
        