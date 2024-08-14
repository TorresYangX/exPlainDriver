import os
import json
import numpy as np
import pandas as pd
from pgm.PGM import PGM
from tqdm import tqdm
from pgm.config import *
from utils import Llama3_map_action, gpt_map_action, update_action, LLM_compare_action
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
        LLM_action = gpt_map_action(LLM_action_)
        ground_action = gpt_map_action(ground_action_)
        LLM_action = update_action(action_list, LLM_action)
        ground_action = update_action(action_list, ground_action)
        if LLM_action == ground_action:
            correct += 1
        else:
            wrong_case.append({
                'id': id,
                'LLM_action': LLM_action_,
                'ground_action': ground_action_
            })            
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
    correct_pgm = 0
    correct = 0
    wrong_case = []
    
    for item in tqdm(LLM_result):
        id = item['image_id']
        LLM_action_ = item['caption']
        ground_action_, _ = id2action(ground_truth_data, id)
        if ground_action_ is None:
            continue
        total += 1
        LLM_action_ = LLM_action_.lower().strip()
        ground_action_ = ground_action_.lower().strip()
        ori_LLM_action = gpt_map_action(LLM_action_)
        ori_LLM_action = update_action(action_list, ori_LLM_action)
        ground_action = gpt_map_action(ground_action_)
        ground_action = update_action(action_list, ground_action)
        if ori_LLM_action == ground_action:
            correct += 1
        acc = correct / total
        
        yolo_class = None
        for item in yolo_detect_Test:
            if item['id'] == id:
                yolo_class = item['classes']
                break
        instance = {
            "action": ori_LLM_action,
            "classes": yolo_class
        }    
        instance_vector = np.array(segment_to_vector(instance))
        prob = pgm.compute_instance_probability(instance_vector)
        condition = instance_vector[config.action_num:]
        if all(x==0 for x in condition):
            LLM_action = ori_LLM_action
        else:
            if prob < 0.001:
                action_probs, index = pgm.infer_action_probability(condition)
                LLM_action = action_list[index]
            else:
                LLM_action = ori_LLM_action
        
        if LLM_action == ground_action:
            correct_pgm += 1
        else:
            info = {
                'ori_LLM_action': ori_LLM_action,
                'LLM_action': LLM_action,
                'ground_action': ground_action,
                'yolo_class': yolo_class
            }
            print(info)
            wrong_case.append(info)            
        acc_pgm = correct_pgm / total
        print(f"Total: {total}, PGM Accuracy: {acc_pgm}, Acc: {acc}")
    with open(wrong_case_save_path, 'w') as f:
        json.dump(wrong_case, f)
    return 0
        
        
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
        LLM_ans = LLM_compare_action(ground_action_, LLM_action_).lower().strip()
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
       
            


if __name__ == "__main__":
    weight_save_path = 'optimal_weights.npy'
    yolo_detect_Test_path = 'process_data/test/test_detected_classes.json'
    LLM_result_path = 'Data/video_process/BDDX_Test_pred_action_vanilla.json'
    ground_truth_path = 'process_data/test/map_ann_test.json'
    wrong_case_file = 'test_case/wrong_case_vanilla_PGM.json'
    # acc = LLM_acc(LLM_result_path, ground_truth_path, wrong_case_file)
    LLM_PGM_acc(weight_save_path, BDDX(), LLM_result_path, ground_truth_path, yolo_detect_Test_path, wrong_case_file)
    
        