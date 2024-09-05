import os
import json
import numpy as np
import pandas as pd
from pgm.PGM import PGM
from tqdm import tqdm
from pgm.config import *
from pgm.predicate_map import segment_to_vector

action_list = ["Keep", "Accelerate", "Decelerate", "Stop", "Reverse", "MakeLeftTurn", 
               "MakeRightTurn", "MakeUTurn", "Merge", "LeftPass", "RightPass", "Yield", 
               "ChangeToLeftLane", "ChangeToRightLane", "Park", "PullOver"]


def id2action(yolo_detect_Test, id):
    for item in yolo_detect_Test:
        if item['id'] == id:
            return item['action'], item['classes'], item['velocity_predicate'], item['direction_predicate']

def is_subset(list1, list2): # if list1 is subset of list2
    return all(elem in list2 for elem in list1)

def predicate_judge(action_list, ground_action_list):
    return is_subset(action_list, ground_action_list)


def LLM_PGM_acc(weights, config, detection_result):
    pgm = PGM(weights=weights, config=config)
    yolo_detect_Test = json.load(open(detection_result)) 
    model_type = "kl"
    LLM_result_predicate = json.load(open(f'result/{model_type}/LLM_result_predicate_{model_type}.json'))
    base_predicate = json.load(open(f'result/vanilla/LLM_result_predicate_vanilla.json'))
    total = 0
    correct_pgm = 0
    correct = 0
    pgm_correct_case = []
    pgm_wrong_case = []
    wrong_case = []
    baseline_correct_model_wrong = []
    for item in tqdm(LLM_result_predicate):
        id = item['id']
        try:
            ground_action, yolo_class, v_p, d_p = id2action(yolo_detect_Test, id)
        except:
            ground_action = None
            continue
        total += 1
        for b_item in base_predicate:
            if b_item['id'] == id:
                basline_LLM_action = b_item['predicate']
                break
        for p_item in LLM_result_predicate:
            if p_item['id'] == id:
                ori_LLM_action = p_item['predicate']
                break
        vailla_flag = False
        pgm_flag = False
        baseline_flag = False
        baseline_flag = predicate_judge(basline_LLM_action, ground_action)
        if predicate_judge(ori_LLM_action, ground_action):
            correct += 1
            vailla_flag = True
        
        acc = correct / total
        
        LLM_action = []
        # judge action in action_list separately
        for action in ori_LLM_action:
            instance = {
                "action": [action],
                "classes": yolo_class,
                'velocity_predicate': v_p,
                'direction_predicate': d_p
            }    
            instance_vector = np.array(segment_to_vector(instance))
            condition = instance_vector[config.action_num:]
            prob = pgm.compute_instance_probability(instance_vector)
            if prob < 0.001:
                _, index = pgm.infer_action_probability(condition)
                LLM_action.append(action_list[index])
            else:
                LLM_action.append(action)
        if predicate_judge(LLM_action, ground_action):
            correct_pgm += 1
            pgm_flag = True
        
        if not vailla_flag and pgm_flag:
            pgm_correct_case.append({
                'id': id,
                'LLM_action': ori_LLM_action,
                'pgm_action': LLM_action,
                'ground_action': ground_action,
                'classes': yolo_class,
            })
        if not pgm_flag and vailla_flag:
            pgm_wrong_case.append({
                'id': id,
                'LLM_action': ori_LLM_action,
                'pgm_action': LLM_action,
                'ground_action': ground_action,
                'classes': yolo_class,
            })
        
        if not pgm_flag and not vailla_flag:
            wrong_case.append({
                'id': id,
                'LLM_action': ori_LLM_action,
                'pgm_action': LLM_action,
                'ground_action': ground_action,
                'classes': yolo_class,
            })
        
        if not pgm_flag and not vailla_flag and baseline_flag:
            baseline_correct_model_wrong.append({
                'id': id,
                'Baseline_LLM_action': basline_LLM_action,
                'LLM_action': ori_LLM_action,
                'pgm_action': LLM_action,
                'ground_action': ground_action,
                'classes': yolo_class,
            })
        
    with open(f'result/{model_type}/bc_mw.json', 'w') as f:
        json.dump(baseline_correct_model_wrong, f)
    with open(f'result/{model_type}/pgm_correct_case.json', 'w') as f:
        json.dump(pgm_correct_case, f)
    with open(f'result/{model_type}/pgm_wrong_case.json', 'w') as f:
        json.dump(pgm_wrong_case, f)
    with open(f'result/{model_type}/wrong_case.json', 'w') as f:
        json.dump(wrong_case, f)
    
    acc_pgm = correct_pgm / total
    print(f"Total: {total}, PGM Accuracy: {acc_pgm}, Acc: {acc}")
    return 0


if __name__ == "__main__":
    yolo_detect_Test_path = 'process_data/test/test_detected_classes.json'
    ground_truth_path = 'process_data/test/map_ann_test.json'
    wrong_case_file = 'test_case/wrong_case_vanilla_PGM.json'
    
    weights = np.array([
        15.51559968,  15.07370472, 15.73000304, 15.45839886,  12.28816603,  12.93708945,
        12.76099285,  12.08004622,  12.59233605,  12.06403711,  12.06403630,  12.03201807,
        12.65637727,  12.51229441,  12.08004540,  12.19210990, 30.65301903, 12.03725726,
        15.06927711, 15.06927711, 15.06927711, 15.06927711, 15.06927711, 
        34.89316792,
        # 34.66902898, 
        # 15.06927711, 
        15.06927711, 
        # 30.89316803, 
        5.00000000, 5.00000000,
        5.00000000, 5.00000000, 
        10.00000000,
        10.00000000, 10.00000000
    ]) 
    LLM_PGM_acc(weights, BDDX(), yolo_detect_Test_path)
    
        