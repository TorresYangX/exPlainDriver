import os
import json
import torch
import numpy as np
import pandas as pd
from pgm.PGM import PGM
from tqdm import tqdm
from utils import Llama3_map_action, update_action
from pgm.YOLO_detector import detect_single_frame
from pgm.predicate_map import segment_to_vector, map_classes_to_vector

action_list = ["Keep", "Accelerate", "Decelerate", "Stop", "Reverse", "MakeLeftTurn", "MakeRightTurn", "MakeUTurn", "Merge", "LeftPass", "RightPass", "Yield", "ChangeToLeftLane", "ChangeToRightLane", "ChangeToCenterLeftTurnLane", "Park", "PullOver"]


def id2action(data, id):
    
    for item in data:
        if item["id"] == id:
            return item['action'], item
    return None, None



def LLM_acc(LLM_result_path, ground_truth_info):
    LLM_result = json.load(open(LLM_result_path))
    ground_truth_data = json.load(open(ground_truth_info))
    
    special_case = []
    
    total = 0
    correct = 0
    for item in tqdm(LLM_result):
        id = item['image_id']
        LLM_action_ = item['caption']
        ground_action_, _ = id2action(ground_truth_data, id)
        if ground_action_ is None:
            continue
        
        total += 1
        LLM_action_ = LLM_action_.lower().strip()
        ground_action_ = ground_action_.lower().strip()
        if LLM_action_ == ground_action_:
            correct += 1
            continue
        
        LLM_action = Llama3_map_action(LLM_action_)
        ground_action = Llama3_map_action(ground_action_)
        
        LLM_action = update_action(action_list, LLM_action)
        ground_action = update_action(action_list, ground_action)
        
        if LLM_action == ground_action:
            correct += 1
        
        # Check for special cases
        if (LLM_action, ground_action) in [("Keep", "Accelerate"), ("Accelerate", "Keep"), ("Stop", "Decelerate"), ("Decelerate", "Stop")]:
            special_case.append({
                'LLM_origin_action': LLM_action_,
                'ground_origin_action': ground_action_,
                'LLM_map_action': LLM_action,
                'ground_map_action': ground_action
            })
            correct += 1
            
    acc = correct / total
    
    print(f"Total: {total}, Correct: {correct}, Accuracy: {acc}")
    
    return acc, special_case


def LLM_PGM_acc(weight_path, LLM_result_path, ground_truth_info, detection_result, save_redress_path):
    pgm = PGM(weight_path=weight_path)
    LLM_result = json.load(open(LLM_result_path))
    ground_truth_data = json.load(open(ground_truth_info))
    yolo_detect_Test = json.load(open(detection_result))
    
    correct_prob = []
    wrong_prob = []
    test_num = 100

    if os.path.exists(save_redress_path):
        os.remove(save_redress_path)
    
    total = 0
    correct = 0
    
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
        ground_action = Llama3_map_action(ground_action_)
        LLM_action = update_action(action_list, LLM_action)
        ground_action = update_action(action_list, ground_action)
        
        
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
        instance_prob = pgm.compute_instance_probability(instance_vector)
        if violate_rule:
            wrong_prob.append(instance_prob)
        else:
            correct_prob.append(instance_prob)
        
        if i == test_num:
            break
    
    correct_prob = np.array(correct_prob)
    wrong_prob = np.array(wrong_prob)
    
    np.save('correct_prob.npy', correct_prob)
    np.save('wrong_prob.npy', wrong_prob)
        
        # if LLM_action_ == ground_action_:
        #     correct += 1
        #     continue
        
        # LLM_action = Llama3_map(LLM_action_)
        # ground_action = Llama3_map(ground_action_)
        
        # LLM_action = update_action(action_list, LLM_action)
        # ground_action = update_action(action_list, ground_action)
        
        # if LLM_action == ground_action:
        #     correct += 1
            
    #     else:
    #         yolo_class = None
    #         for item in yolo_detect_Test:
    #             if item['id'] == id:
    #                 yolo_class = item['classes']
    #                 print(f"YOLO: {yolo_class}, id: {id}")
    #                 break
    #         intance = {
    #             "action": LLM_action,
    #             "classes": yolo_class
    #         }    
            
    #         instance_vector = np.array(segment_to_vector(intance))
    #         violate_rule = pgm.validate_instance(instance_vector)
    #         if violate_rule:
                
    #             violate_info = {
    #                 'video_info': video_info,
    #                 'yolo_class': yolo_class,
    #                 'LLM_action': LLM_action,
    #                 'ground_truth_action': ground_action,
    #                 'violate_rule': violate_rule
    #             }
                
    #             redress_case.append(violate_info)

    #             with open(save_redress_path, 'a') as f:
    #                 json.dump(violate_info, f)
    #                 f.write(',\n')
                    
    # return correct / total, redress_case
                
            
        
        # Check for special cases
    #     elif (LLM_action, ground_action) in [("Keep", "Accelerate"), ("Accelerate", "Keep"), ("Stop", "Decelerate"), ("Decelerate", "Stop")]:
    #         correct += 1
            
    #     else:
    #         yolo_class = None
    #         for item in yolo_detect_Test:
    #             if item['id'] == id:
    #                 yolo_class = item['classes']
    #                 print(f"YOLO: {yolo_class}, id: {id}")
    #                 break
    #         condition_vector = np.array(map_classes_to_vector(yolo_class))[len(PGM_action_list):]
    #         _, action_index = pgm.infer_action_probability(condition_vector)
    #         PGM_LLM_action = PGM_action_list[action_index]
    #         print(f"LLM: {PGM_LLM_action}, Ground Truth: {ground_action}")
    #         if PGM_LLM_action == ground_action:
    #             correct += 1
    #         elif (PGM_LLM_action, ground_action) in [("Keep", "Accelerate"), ("Accelerate", "Keep"), ("Stop", "Decelerate"), ("Decelerate", "Stop")]:
    #             correct += 1    
    #         else:
    #             continue
            
    # acc = correct / total
    
    # print(f"Total: {total}, Correct: {correct}, Accuracy: {acc}")
    
    # return acc
    



if __name__ == "__main__":
    LLM_result_path = 'Data/video_process/BDDX_Test_pred_action_weight0.0.json'
    ground_truth_path = 'process_data/test/map_ann_test.json'
    weight_save_path = 'optimal_weights_Interface.npy'
    yolo_detect_Test_path = 'process_data/test/test_detected_classes.json'
    
    save_redress_path = 'test_case/redress_case_0.0.json'
    
    LLM_PGM_acc(weight_save_path, LLM_result_path, ground_truth_path, yolo_detect_Test_path, save_redress_path)
    
    # acc, special_case = LLM_acc(LLM_result_path, ground_truth_path)
    
        