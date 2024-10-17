import re 
import json
import numpy as np
from pgm.config import DriveLM
from pgm.PGM_drivelm import PGM
from pgm.predicate_map_drivelm import segment_to_vector
from utils_drivelm import action_map
from pgm.config import DriveLM

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

def splitSpeedCourceDescription(description):
    speed_desps = ['driving fast', 'driving very fast', 'driving slowly', 'driving with normal speed', 'not moving']
    cource_desps = ['slightly steering to the left', 'slightly steering to the right', 'steering to the left', 'steering to the right', 'going straight']
    for speed_desp in speed_desps:
        if speed_desp in description:
            speed = speed_desp
            break
    for cource_desp in cource_desps:
        if cource_desp in description:
            cource = cource_desp
            break
    return speed, cource


def DriveLM_Test(weight, detect_save_path, question_path, llm_predicate_path, llm_prediction_path):
    with open(detect_save_path) as f:
        detect_data = json.load(f)
    # drivelm_question_file
    with open(question_path) as f:
        question_data = json.load(f)
    # pred_predicate_file
    with open(llm_predicate_path) as f:
        pred_data = json.load(f)
    # pred_origin_file
    with open(llm_prediction_path) as f:
        origin_data = json.load(f)
    # ground_answer_file
    with open('Data/DriveLM/DriveLM_process/drivelm_val_clean.json') as f:
        answer_data = json.load(f)    
    pgm = PGM(weights = weight, config=DriveLM())
    
    ori_correct_count = 0
    pred_correct_count = 0
    ori_speed_acc = 0
    ori_course_acc = 0
    pred_speed_acc = 0
    pred_course_acc = 0
    
    for item in pred_data:
        image_id = item['image_id']
        scene_id = image_id.split('_')[0]
        keyframe_id = image_id.split('_')[1]
        
        question = question_data[scene_id]["key_frames"][keyframe_id]["QA"]["behavior"][0]["Q"]
        option_list = question2option(question)
        option_action_list = optionlist2actionlist(option_list)
        ground_ans = answer_data[scene_id]["key_frames"][keyframe_id]["QA"]["behavior"][0]["A"]
        ground_description = option2description(option_list, ground_ans)
        ground_speed_description, ground_cource_description = splitSpeedCourceDescription(ground_description)
        
        ori_pred_ans = id2predAns(image_id, origin_data)
        ori_description = option2description(option_list, ori_pred_ans)
        ori_speed_description, ori_cource_description = splitSpeedCourceDescription(ori_description)
        
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
                velo_action = DriveLM().action_list[velocity_index]
                dire_action = DriveLM().action_list[direction_index + DriveLM().velocity_action_num]
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
            if len(probable_answers) > 1:
                print (f"Multiple probable answers: {probable_answers}, {image_id}")
            pred_ans = probable_answers[0]
        
        pred_description = option2description(option_list, pred_ans)
        pred_speed_description, pred_cource_description = splitSpeedCourceDescription(pred_description)

        if ori_pred_ans == ground_ans:
            ori_correct_count += 1
        if pred_ans == ground_ans:
            pred_correct_count += 1
        
        if ori_speed_description == ground_speed_description:
            ori_speed_acc += 1
        if ori_cource_description == ground_cource_description:
            ori_course_acc += 1
        if pred_speed_description == ground_speed_description:
            pred_speed_acc += 1
        if pred_cource_description == ground_cource_description:
            pred_course_acc += 1    
        
    print (f"Original accuracy: {ori_correct_count / len(pred_data)}, Predicted accuracy: {pred_correct_count / len(pred_data)}")
    print (f"Original speed accuracy: {ori_speed_acc / len(pred_data)}, Original course accuracy: {ori_course_acc / len(pred_data)}")
    print (f"Predicted speed accuracy: {pred_speed_acc / len(pred_data)}, Predicted course accuracy: {pred_course_acc / len(pred_data)}")
                                        
           
# if __name__ == '__main__':
#     weight = np.load('weights/optimal_weights_drivelm_rag_pdce.npy')
#     main(weight)
        
            
        
        
    