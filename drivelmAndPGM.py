import re 
import json
import numpy as np
from pgm.config import DriveLM
from pgm.PGM_drivelm import PGM
from pgm.predicate_map_drivelm import segment_to_vector

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
        '6': ['slightly steering to the left'],
        '7': ['slightly steering to the right'],
        '8': ['going straight']
    }
    return index2action[str(index)]

def question2option(question):
    pattern = r"([A-D])\. (.*?)(?= [A-D]\.|$)"
    matches = re.findall(pattern, question)
    options_list = [(match[0], match[1].strip()) for match in matches]
    return options_list

def action2description(action_list):
    velo_predicate = None
    dire_predicate = None
    for action in action_list:
        if action in ['Keep', 'Accelerate', 'Decelerate', 'Stop']:
            velo_predicate = action
        else:
            dire_predicate = action

    action2description = {
        'Keep': ['driving with normal speed'],
        'Accelerate': ['driving very fast', 'driving fast'],
        'Decelerate': ['driving slowly'],
        'Stop': ['not moving'],
        'MakeLeftTurn': ['steering to the left'],
        'MakeRightTurn': ['steering to the right'],
        'ChangeToLeftLane': ['slightly steering to the left'],
        'ChangeToRightLane': ['slightly steering to the right'],
        'Straight': ['going straight']
    }
    return {'velo_desc': action2description[velo_predicate], 'dire_desc': action2description[dire_predicate]}


def test_pgm():
    with open('process_data_drivelm/train/train_detected_classes.json') as f:
        detect_data = json.load(f)
        
    weight = np.load('optimal_weights_drivelm.npy')
    pgm = PGM(weights = weight, config=DriveLM())
    wrong_items = []
    for item in detect_data:
        action_list = item['action']
        condition_predicate = item['classes']
        segment = {
            'action': action_list,
            'classes': condition_predicate
        }
        vector = np.array(segment_to_vector(segment))
        velo_prob, dire_prob = pgm.compute_instance_probability(vector)
        if velo_prob < 1e-3 or dire_prob < 1e-3:
            wrong_item = {
                'image_id': item['image_id'],
                'velo_prob': velo_prob,
                'dire_prob': dire_prob,
                'action': action_list,
                'classes': condition_predicate
            }
            wrong_items.append(wrong_item)
        with open('drivelm_wrong_items.json', 'w') as f:
            json.dump(wrong_items, f, indent=4)


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
    
    weight = np.load('optimal_weights_drivelm.npy')
    pgm = PGM(weights = weight, config=DriveLM())
    
    for item in pred_data:
        image_id = item['image_id']
        action_list = item['action_list']
        detect_item = map_detect_item(image_id, detect_data)
        condition_predicate = detect_item['classes']
        segment = {
            'action': action_list,
            'classes': condition_predicate
        }
        vector = np.array(segment_to_vector(segment))
        condition_vector = vector[DriveLM().action_num:]
        velo_prob, dire_prob = pgm.compute_instance_probability(vector)
        # for dire_predicate in ['SINGLE_SOLID_WHITE_LEFT','SINGLE_SOLID_WHITE_RIGHT']:
        #     if dire_predicate in condition_predicate and 'Straight' not in action_list:
        #         print('action_list:', action_list, 'condition_predicate:', condition_predicate)
        #         print('velo_prob:', velo_prob, 'dire_prob:', dire_prob)
        #         print(vector)
        new_velo_descs = None
        new_dire_descs = None
        if velo_prob < 1e-2:
           _, velo_index, _ = pgm.infer_action_probability(condition_vector)
           new_velo_descs = map_index_to_description(velo_index)
        if dire_prob < 1e-2:
           _, _, dire_index = pgm.infer_action_probability(condition_vector)
           new_dire_descs = map_index_to_description(dire_index)
        if new_velo_descs or new_dire_descs:
            scene_id = image_id.split('_')[0]
            keyframe_id = image_id.split('_')[1]
            question = question_data[scene_id]["key_frames"][keyframe_id]["QA"]["behavior"][0]["Q"]
            option_list = question2option(question)
            if new_velo_descs and not new_dire_descs:
                final_velo_descs = new_velo_descs
                final_dire_descs = action2description(action_list)['dire_desc']
            elif new_dire_descs and not new_velo_descs:
                final_dire_descs = new_dire_descs
                final_velo_descs = action2description(action_list)['velo_desc']
            else:
                final_velo_descs = new_velo_descs
                final_dire_descs = new_dire_descs
            modify_option = []
            possible_options = []
            for option_item in option_list:
                for desc in final_velo_descs:
                    if desc in option_item[1]:
                        possible_options.append(option_item)
            for possible_option in possible_options:
                for desc in final_dire_descs:
                    if desc in possible_option[1]:
                        modify_option.append(possible_option[0])
            if len(modify_option) != 1:
                print('error detect:', image_id)
                print('error')
                print('classes:', condition_predicate , 'final_velo_descs:', final_velo_descs, 'final_dire_descs:', final_dire_descs, 'option_list:', option_list, 'modify_option:', modify_option, 'possible_options:', possible_options)
            else:
                print('error detect:', image_id)
                print('modify_option:', modify_option)            
            

           
if __name__ == '__main__':
    main()
    # test_pgm()
        
            
        
        
    