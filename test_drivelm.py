import re 
import json
import random
import numpy as np
from pgm.config import DriveLM
from pgm.PGM_drivelm import PGM
from pgm.predicate_map_drivelm import segment_to_vector
from utils_drivelm import action_map

all_action_list = ['Normal', 'Fast', 'Slow', 'Stop', 'Left', 'Right', 'Straight']

def map_detect_item(id, detect_data):
    for item in detect_data:
        if item['image_id'] == id:
            return item
    return None


def question2option(question):
    pattern = r"([A-D])\. (.*?)(?= [A-D]\.|$)"
    matches = re.findall(pattern, question)
    options_list = [(match[0], match[1].strip()) for match in matches]
    return options_list

def option2description(option_list, option):
    for item in option_list:
        if item[0] == option:
            return item[1]

def optionlist2actionlist(option_list):
    option_action_list = []
    for item in option_list:
        option_action_list.append((item[0],set(action_map(item[1]))))
    return option_action_list

def id2predAns(id, pred_data):
    for item in pred_data:
        if id == item[0]['image_id']:
            return item[-2]['caption']
    return None


def main(weight):
    pattern = 'vanilla'
    # drivelm_detect_file
    with open('process_data_drivelm/test/test_detected_classes.json') as f:
        detect_data = json.load(f)
    # drivelm_question_file
    with open('DriveLM_process/v1_1_val_nus_q_only.json') as f:
        question_data = json.load(f)
    # pred_predicate_file
    with open(f'result/drivelm_{pattern}/LLM_result.json') as f:
        pred_data = json.load(f)
    # pred_origin_file
    with open('DriveLM_process/DrivingLM_Test_pred.json') as f:
        origin_data = json.load(f)
    # ground_answer_file
    with open('DriveLM_process/drivelm_val_clean.json') as f:
        answer_data = json.load(f)    

    # weight = np.load('optimal_weights_drivelm_llm_rule.npy')
    pgm = PGM(weights = weight, config=DriveLM())
    
    ori_correct_count = 0
    pred_correct_count = 0
    undetect_items = []
    misdetect_items = []
    corrdetect_items = []
    
    for item in pred_data:
        image_id = item['image_id']
        scene_id = image_id.split('_')[0]
        keyframe_id = image_id.split('_')[1]
        
        question = question_data[scene_id]["key_frames"][keyframe_id]["QA"]["behavior"][0]["Q"]
        option_list = question2option(question)
        option_action_list = optionlist2actionlist(option_list)
        ground_ans = answer_data[scene_id]["key_frames"][keyframe_id]["QA"]["behavior"][0]["A"]
        ground_action_list = action_map(option2description(option_list, ground_ans))
        
        ori_pred_ans = id2predAns(image_id, origin_data)
        
        action_list = item['action']
        detect_item = map_detect_item(image_id, detect_data)
        condition_predicate = detect_item['classes']
        velo_predicate = detect_item['velocity_predicate']
        dire_predicate = detect_item['direction_predicate']
        segment = {
            'action': action_list,
            'classes': condition_predicate,
            'velocity_predicate': velo_predicate,
            'direction_predicate': dire_predicate
        }
        vector = np.array(segment_to_vector(segment, action_list))
        condition_vector = vector[DriveLM().action_num:]
        velo_probs, dire_probs, _, _ = pgm.infer_action_probability(condition_vector)
        velo_probs_index = np.argsort(-velo_probs)
        dire_probs_index = np.argsort(-dire_probs)
        probable_answers = []
        found = False
        for direction_index in dire_probs_index:
            for velocity_index in velo_probs_index:
                velo_action = all_action_list[velocity_index]
                dire_action = all_action_list[direction_index + DriveLM().velocity_action_num]
                action_set = set([velo_action, dire_action])
                for option_item in option_action_list:
                    if action_set == option_item[1]:
                        probable_answers.append(option_item[0])
                        found = True
                if found:
                    break
            if found:
                break
        
        if ori_pred_ans in probable_answers:
            pred_ans = ori_pred_ans
        else:
            pred_ans = probable_answers[0]

        if ori_pred_ans == ground_ans:
            ori_correct_count += 1
        if pred_ans == ground_ans:
            pred_correct_count += 1
        if ori_pred_ans == ground_ans and pred_ans != ground_ans:
            misdetect_items.append({
                'image_id': image_id,
                'ground_ans': ground_ans,
                'ori_pred_ans': ori_pred_ans,
                'pred_ans': pred_ans,
                'ground_action': ground_action_list,
                'pred_action': action_list,
                'modify_action': list(action_set),
                'condition_predicate': condition_predicate,
                'velocity_predicate': velo_predicate,
                'direction_predicate': dire_predicate,
                'option_action_list': [(item[0], list(item[1])) for item in option_action_list]
            })
        if ori_pred_ans != ground_ans and pred_ans == ground_ans:
            corrdetect_items.append({
                'image_id': image_id,
                'ground_ans': ground_ans,
                'ori_pred_ans': ori_pred_ans,
                'pred_ans': pred_ans,
                'ground_action': ground_action_list,
                'pred_action': action_list,
                'modify_action': list(action_set),
                'condition_predicate': condition_predicate,
                'velocity_predicate': velo_predicate,
                'direction_predicate': dire_predicate,
                'option_action_list': [(item[0], list(item[1])) for item in option_action_list]
            })
        if ori_pred_ans != ground_ans and pred_ans != ground_ans:  
            undetect_items.append({
                'image_id': image_id,
                'ground_ans': ground_ans,
                'ori_pred_ans': ori_pred_ans,
                'pred_ans': pred_ans,
                'ground_action': ground_action_list,
                'pred_action': action_list,
                'modify_action': list(action_set),
                'condition_predicate': condition_predicate,
                'velocity_predicate': velo_predicate,
                'direction_predicate': dire_predicate,
                'option_action_list': [(item[0], list(item[1])) for item in option_action_list]
            })
        
    print (f"Original accuracy: {ori_correct_count / len(pred_data)}")
    print (f"Predicted accuracy: {pred_correct_count / len(pred_data)}")
    # save the result
    with open(f'result/drivelm_{pattern}/undetect_items.json', 'w') as f:
        json.dump(undetect_items, f)
    with open(f'result/drivelm_{pattern}/misdetect_items.json', 'w') as f:
        json.dump(misdetect_items, f)
    with open(f'result/drivelm_{pattern}/corrdetect_items.json', 'w') as f:
        json.dump(corrdetect_items, f)
                                        
           
if __name__ == '__main__':
    weight = np.array([ 3.99999996,  4.74662582,  4.26896812,  4.68809649,  3.53423538,  3.75047566, 5.59472847, 
                    
                    17.86671967, 17.8794396,  17.8794396,  17.8794396,  17.8794396,
                    17.8794396,  17.85823962, 17.8794396,  17.8794396,  17.8794396,  17.8794396,
                    
                    17.70061279, 19.52688565, 19.58688286, 19.49360417, 13.65220948, 13.56018218, 13.40880087, 
                    17.8794396,  17.8794396,  17.8794396,  17.8794396,  17.8794396, 17.8794396,  17.8794396 ])
    main(weight)
        
            
        
        
    