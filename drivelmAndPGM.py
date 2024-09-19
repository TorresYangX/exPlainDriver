import re 
import json
import random
import numpy as np
from pgm.config import DriveLM
from pgm.PGM_drivelm import PGM
from pgm.predicate_map_drivelm import segment_to_vector
from utils_drivelm import action_map

def map_detect_item(id, detect_data):
    for item in detect_data:
        if item['image_id'] == id:
            return item
    return None

def map_index_to_description(index):
    index2action = {
        '0': ['driving with normal speed'],
        '1': ['driving very fast', 'driving fast'],
        '2': ['driving slowly'],
        '3': ['not moving'],
        '4': ['steering to the left'],
        '5': ['steering to the right'],
        '6': ['going straight', 'slightly steering to the left', 'slightly steering to the right'],
    }
    return index2action[str(index)]

def question2option(question):
    pattern = r"([A-D])\. (.*?)(?= [A-D]\.|$)"
    matches = re.findall(pattern, question)
    options_list = [(match[0], match[1].strip()) for match in matches]
    return options_list

def option2description(option_list, option):
    for item in option_list:
        if item[0] == option:
            return item[1]

def action2description(action_list):
    velo_predicate = None
    dire_predicate = None
    for action in action_list:
        if action in ['Normal', 'Fast', 'Slow', 'Stop']:
            velo_predicate = action
        else:
            dire_predicate = action
    action2description = {
        'Normal': ['driving with normal speed'],
        'Fast': ['driving very fast', 'driving fast'],
        'Slow': ['driving slowly'],
        'Stop': ['not moving'],
        'Left': ['steering to the left'],
        'Right': ['steering to the right'],
        'Straight': ['going straight', 'slightly steering to the left', 'slightly steering to the right']
    }
    return {'velo_desc': action2description[velo_predicate], 'dire_desc': action2description[dire_predicate]}

def desc_filter_option(desc, option_list):
    probable_option = []
    for opt in option_list:
        for sub_desc in desc:
            if sub_desc in opt[1]:
                probable_option.append(opt)
                break
    return probable_option

def id2predAns(id, pred_data):
    for item in pred_data:
        if id == item[0]['image_id']:
            return item[-2]['caption']
    return None


def main():
    # drivelm_detect_file
    with open('process_data_drivelm/test/test_detected_classes.json') as f:
        detect_data = json.load(f)
    # drivelm_question_file
    with open('DriveLM_process/v1_1_val_nus_q_only.json') as f:
        question_data = json.load(f)
    # pred_predicate_file
    with open('result/drivelm/LLM_result_predicate.json') as f:
        pred_data = json.load(f)
    # pred_origin_file
    with open('DriveLM_process/DrivingLM_Test_pred.json') as f:
        origin_data = json.load(f)
    # ground_answer_file
    with open('DriveLM_process/drivelm_val_clean.json') as f:
        answer_data = json.load(f)    

    weight = np.load('optimal_weights_drivelm.npy')
    pgm = PGM(weights = weight, config=DriveLM())
    
    undetect_items = []
    misdetect_items = []
    corrdetect_items = []
    
    for item in pred_data:
        image_id = item['image_id']
        scene_id = image_id.split('_')[0]
        keyframe_id = image_id.split('_')[1]
        
        question = question_data[scene_id]["key_frames"][keyframe_id]["QA"]["behavior"][0]["Q"]
        option_list = question2option(question)
        ground_ans = answer_data[scene_id]["key_frames"][keyframe_id]["QA"]["behavior"][0]["A"]
        ground_action_list = action_map(option2description(option_list, ground_ans))
        pred_ans = id2predAns(image_id, origin_data)
        
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
        vector = np.array(segment_to_vector(segment))
        condition_vector = vector[DriveLM().action_num:]
        velo_prob, dire_prob = pgm.compute_instance_probability(vector)
        velo_probs = None
        dire_probs = None
        modify_ans = None
        if velo_prob < 1e-2:
           velo_probs, _, _, _ = pgm.infer_action_probability(condition_vector)
        if dire_prob < 1e-2:
           _, dire_probs, _, _ = pgm.infer_action_probability(condition_vector)
        if velo_probs is not None or dire_probs is not None:
            if velo_probs is None and dire_probs is not None:
                velo_desc = action2description(action_list)['velo_desc']
                probable_option = desc_filter_option(velo_desc, option_list)
                dire_probs = np.array(dire_probs)
                dire_probs_index = dire_probs.argsort()[::-1] + DriveLM().velocity_action_num
                for index in dire_probs_index:
                    dire_desc = map_index_to_description(index)
                    final_option = desc_filter_option(dire_desc, probable_option)
                    if len(final_option) > 0:
                        break
            elif dire_probs is None and velo_probs is not None:
                dire_desc = action2description(action_list)['dire_desc']
                probable_option = desc_filter_option(dire_desc, option_list)
                velo_probs = np.array(velo_probs)
                velo_probs_index = velo_probs.argsort()[::-1]
                for index in velo_probs_index:
                    velo_desc = map_index_to_description(index)
                    final_option = desc_filter_option(velo_desc, probable_option)
                    if len(final_option) > 0:
                        break
            else:
                velo_probs = np.array(velo_probs)
                dire_probs = np.array(dire_probs)
                velo_probs_index = velo_probs.argsort()[::-1]
                dire_probs_index = dire_probs.argsort()[::-1] + DriveLM().velocity_action_num
                for index in velo_probs_index:
                    velo_desc = map_index_to_description(index)
                    probable_option = desc_filter_option(velo_desc, option_list)
                    for index in dire_probs_index:
                        dire_desc = map_index_to_description(index)
                        final_option = desc_filter_option(dire_desc, probable_option)
                        if len(final_option) > 0:
                            break
                    if len(final_option) > 0:
                        break
            final_option_option = [item[0] for item in final_option]
            if pred_ans in final_option_option:
                modify_ans = pred_ans
            else:
                modify_ans = random.choice(final_option_option)
            modify_action_list = action_map(option2description(option_list, modify_ans))
            if modify_ans == ground_ans and pred_ans != ground_ans:
                record_item = {
                    'image_id': image_id,
                    'ground_ans': ground_ans,
                    'pred_ans': pred_ans,
                    'modify_ans': modify_ans,
                    'classes': condition_predicate,
                    'velocity_predicate': velo_predicate,
                    'direction_predicate': dire_predicate,
                    'pred_action': action_list,
                    'ground_action': ground_action_list,
                    'modify_action': modify_action_list
                }
                corrdetect_items.append(record_item)
            elif pred_ans == ground_ans and modify_ans != ground_ans:
                record_item = {
                    'image_id': image_id,
                    'ground_ans': ground_ans,
                    'pred_ans': pred_ans,
                    'modify_ans': modify_ans,
                    'classes': condition_predicate,
                    'velocity_predicate': velo_predicate,
                    'direction_predicate': dire_predicate,
                    'pred_action': action_list,
                    'ground_action': ground_action_list,
                    'modify_action': modify_action_list
                }
                misdetect_items.append(record_item)
        if modify_ans is None and pred_ans != ground_ans:
            record_item = {
                'image_id': image_id,
                'ground_ans': ground_ans,
                'pred_ans': pred_ans,
                'classes': condition_predicate,
                'velocity_predicate': velo_predicate,
                'direction_predicate': dire_predicate,
                'pred_action': action_list,
                'ground_action': ground_action_list,
                'modify_action': modify_action_list
            }
            undetect_items.append(record_item)
    
    # save the result
    with open('result/drivelm/undetect_items.json', 'w') as f:
        json.dump(undetect_items, f)
    with open('result/drivelm/misdetect_items.json', 'w') as f:
        json.dump(misdetect_items, f)
    with open('result/drivelm/corrdetect_items.json', 'w') as f:
        json.dump(corrdetect_items, f)
                                
           
if __name__ == '__main__':
    main()
        
            
        
        
    