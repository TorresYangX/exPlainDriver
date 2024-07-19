import json
import torch
import numpy as np
import pandas as pd
from PGM import PGM
from tqdm import tqdm
from YOLO_detector import detect_single_frame, Llama3_map
from predicate_map import map_classes_to_vector
from transformers import AutoModelForCausalLM, AutoTokenizer

action_list = ["Keep", "Accelerate", "Decelerate", "Stop", "Reverse", "MakeLeftTurn", "MakeRightTurn", "MakeUTurn", "Merge", "LeftPass", "RightPass", "Yield", "ChangeToLeftLane", "ChangeToRightLane", "ChangeToCenterLeftTurnLane", "Park", "PullOver"]

PGM_action_list = ["Decelerate", "Stop", "Reverse", "MakeLeftTurn", "MakeRightTurn", "MakeUTurn", "LeftPass", "RightPass"]

annotation_path = 'Data/BDD-X-Dataset/BDD-X-Annotations_v1.csv'
ori_annotation = pd.read_csv(annotation_path)
ori_annotation = ori_annotation.dropna(subset=['Input.Video'])
ori_annotation['Input.Video'] = ori_annotation['Input.Video'].apply(lambda x: x.split('/')[-1])


def original_video_segment_map(video_name, action):
    row = ori_annotation[ori_annotation['Input.Video'] == video_name].iloc[0].dropna()
    action_columns = [col for col in row.index if 'action' in col]
    for col in action_columns:
        row[col] = row[col].strip().lower()
    action = action.strip().lower()
    target_col = None
    target_num = None
    for col in action_columns:
        if row[col] == action:
            target_col = col
            break
    if target_col:
        target_num = target_col.split('.')[1].split('action')[0]
        
    if target_num==None:
        return None, None
        
    start_col = 'Answer.'+target_num+'start'
    end_col = 'Answer.'+target_num+'end'
    try:
        start_time = row[start_col]
        end_time = row[end_col]
        return start_time, end_time
    except Exception as e:
        print(f"Error: {video_name}, {action}. Exception: {e}")
        return None, None


def id2action(data, id):
    action = None
    video_info = {}
    
    for item in data:
        if item["id"] == id:
            original_video = item['video'].split('_')[-2].split('/')[-1]+'.mov'
            action = next(convo['value'] for convo in item['conversations'] if convo['from'] == 'gpt' and 'What is the action of ego car?' in item['conversations'][0]['value'])
            start_time, end_time = original_video_segment_map(original_video, action)
            video_info={
                'original_video': "Data/BDD-X/Videos/videos/"+original_video,
                'start_time': start_time,
                'end_time': end_time,
                'process_video': item['video'],
                'action': action,
            }
            break
            
    return action, video_info


def update_action(action_list, action):
    for act in action_list:
        if act.lower() in action.lower():
            return act
    return action



def LLM_acc(LLM_result_path, ground_truth_path):
    LLM_result = json.load(open(LLM_result_path))
    ground_truth_data = json.load(open(ground_truth_path))
    
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
        
        LLM_action = Llama3_map(LLM_action_)
        ground_action = Llama3_map(ground_action_)
        
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


def LLM_PGM_acc(weight_save_path, LLM_result_path, ground_truth_path):
    pgm = PGM(weight_path=weight_save_path)
    
    LLM_result = json.load(open(LLM_result_path))
    ground_truth_data = json.load(open(ground_truth_path))
    
    special_case = []
    wrong_case = []
    
    correct_case = []
    
    total = 0
    correct = 0
    
    for item in tqdm(LLM_result):
        id = item['image_id']
        LLM_action_ = item['caption']
        ground_action_, video_info = id2action(ground_truth_data, id)
        
        if ground_action_ is None:
            continue
        
        total += 1
        LLM_action_ = LLM_action_.lower().strip()
        ground_action_ = ground_action_.lower().strip()
        if LLM_action_ == ground_action_:
            correct += 1
            continue
        
        LLM_action = Llama3_map(LLM_action_)
        ground_action = Llama3_map(ground_action_)
        
        LLM_action = update_action(action_list, LLM_action)
        ground_action = update_action(action_list, ground_action)
        
        if LLM_action == ground_action:
            correct += 1
        
        # Check for special cases
        elif (LLM_action, ground_action) in [("Keep", "Accelerate"), ("Accelerate", "Keep"), ("Stop", "Decelerate"), ("Decelerate", "Stop")]:
            special_case.append({
                'LLM_origin_action': LLM_action_,
                'ground_origin_action': ground_action_,
                'LLM_map_action': LLM_action,
                'ground_map_action': ground_action
            })
            correct += 1
            
        else:
            yolo_class = detect_single_frame(video_info)
            print(f"YOLO: {yolo_class}")
            condition_vector = np.array(map_classes_to_vector(yolo_class))[len(PGM_action_list):]
            _, action_index = pgm.infer_action_probability(condition_vector)
            PGM_LLM_action = PGM_action_list[action_index]
            print(f"LLM: {PGM_LLM_action}, Ground Truth: {ground_action}")
            if PGM_LLM_action == ground_action:
                correct += 1
            elif (PGM_LLM_action, ground_action) in [("Keep", "Accelerate"), ("Accelerate", "Keep"), ("Stop", "Decelerate"), ("Decelerate", "Stop")]:
                correct_case.append({
                    'video_info': video_info,
                    'yolo_class': yolo_class,
                    'LLM_action': LLM_action,
                    'LLM_PGM_action': PGM_LLM_action,
                    'ground_truth_action': ground_action  
                })
                correct += 1
                # save the correct case
                with open('correct_case.json', 'w') as f:
                    json.dump(correct_case, f)
                
            else:
                wrong_case.append({
                  'video': video_info,
                  'yolo_class': yolo_class,
                  'LLM_PGM_action': PGM_LLM_action,
                  'ground_truth_action': ground_action  
                })
                
                    
            
    acc = correct / total
    
    print(f"Total: {total}, Correct: {correct}, Accuracy: {acc}")
    
    return acc, special_case, wrong_case
    
    
    

if __name__ == "__main__":
    LLM_result_path = 'Data/video_process/BDDX_Test_pred_action_weight0.0.json'
    ground_truth_path = 'Data/video_process/conversation_bddx_test.json'
    weight_save_path = 'optimal_weights.npy'
    # acc, special_case = LLM_acc(LLM_result_path, ground_truth_path)

    acc, special_case, wrong_case = LLM_PGM_acc(weight_save_path, LLM_result_path, ground_truth_path)
    
        