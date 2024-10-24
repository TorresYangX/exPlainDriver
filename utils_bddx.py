import re
import os
import json
import string
import pickle
import numpy as np
from tqdm import tqdm
from pgm.PGM import PGM
from openai import OpenAI
from pgm.predicate_map import json_to_vectors
from pgm.video_annotation import query_annotation_csv
# from pgm.config import BDDX
from arixv.MLN_sole.config import BDDX
import random

def gpt_map_action(action):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    system_prompt = "You are a helpful assistant"
    prompt = """Given the current behavior of the car, please use one or two predicates below to best describe the behavior of the car. The predicates are: 
    Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass, Yield, ChangeToLeftLane, ChangeToRightLane, Park, PullOver.
    Here are some examples:
    #Current Behavior#: The car is travelling down the road.
    #Predicates#: Keep\n
    #Current Behavior#: The car is making left turn.
    #Predicates#: MakeLeftTurn\n
    #Current Behavior#: The car is slowing down and then comes to a stop.
    #Predicates#: Decelerate, Stop\n
    #Current Behavior#: The car is accelerating and then turns right.
    #Predicates#: Accelerate, MakeRightTurn\n
    #Current Behavior#: The car is making a left turn and accelerates.
    #Predicates#: MakeLeftTurn, Accelerate\n
    #Current Behavior#: The car decelerates and stops.
    #Predicates#: Decelerate, Stop\n
    
    Now the current behavior of the car is described, provide the predicates that best describe the behavior of the car.:
    
    #Current Behavior#: {action}
    #Predicates#: """.format(action=action)
    
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
                model='gpt-4o',
                messages=messages,
                temperature=0.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            ).choices[0]
    return response.message.content


def gpt_map_cs(Speed, Curvature, Acceleration, Course):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    system_prompt = "You are a helpful assistant"
    prompt = """
    Given the current speed, curvature, acceleration, and course of the car, use one velocity predicate and one directional predicate to best describe the behavior of the car. 
    The velocity predicates are: Keep, Accelerate, Decelerate, Stop, Reverse.
    The directional predicates are: Straight, Left, Right. 
    Output the predicates directly without any additional information.
    Here are some examples:
    # Speed: [7.18, 5.76, 4.45, 3.30, 2.24, 1.20, 0.36]
    # Curvature: [1.32, 0.88, 0.58, 1.85, 2.74, 1.61, 0.64]
    # Acceleration: [-1.22, -1.85, -2.39, -2.22, -2.01, -1.46, -0.87]
    # Course: [0.00, -10.03, -8.33, -3.23, -0.97, -0.32, -0.08]
    # Predicate: Stop, Left
    # Speed: [12.31, 9.51, 7.24, 5.38, 3.67, 2.76, 3.00]
    # Curvature: [-0.00, 0.00, 0.00, -0.05, -0.18, -0.67, -0.79]
    # Acceleration: [-1.85, -2.79, -2.73, -2.23, -1.67, -0.47, 0.71]
    # Course: [0.00, 0.00, 0.00, 0.00, -20.26, -60.78, 7.17]
    # Predicate: Decelerate, Right
    # Speed: [1.27, 4.18, 6.83, 8.87, 10.44, 12.22, 14.45]
    # Curvature: [0.00, 0.00, 0.00, -0.00, -0.01, -0.00, -0.00]
    # Acceleration: [2.27, 2.15, 1.81, 1.35, 1.28, 1.56, 1.45]
    # Course: [0.00, -0.09, 0.00, 0.00, 0.20, 0.00, 0.00]
    # Predicate: Accelerate, Straight
    # Speed: {speed}
    # Curvature: {curvature}
    # Acceleration: {acceleration}
    # Course: {course}
    # Predicate: """.format(speed=Speed, curvature=Curvature, acceleration=Acceleration, course=Course)
    messages=[{"role": "system", "content": system_prompt},
              {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
                model='gpt-4o',
                messages=messages,
                temperature=0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            ).choices[0]
    return response.message.content


def update_action(action_list, action):
    for act in action_list:
        if act.lower() in action.lower():
            return act
    return None


def update_action_set(action_list, ori_actions):
    action_set = []
    for act in action_list:
        if act.lower() in ori_actions.lower():
            action_set.append(act)
    return action_set


def cs_extractor(id, cs_info):
    data = {}
    for item in cs_info:
        if item['id'] == id:
            text = item['conversations'][0]['value']
            pattern = r'(Speed|Curvature|Acceleration|Course): \[([^\]]+)\]'
            matches = re.findall(pattern, text)
            for match in matches:
                key = match[0]
                values = list(map(float, match[1].split(', ')))
                data[key] = values
            break
    return data


def map_LLM_pred(LLM_result_path, save_path):
    action_list=BDDX().action_list
    LLM_result = json.load(open(LLM_result_path))
    extract_result = []
    for item in tqdm(LLM_result):
        id = item['image_id']
        action = item['caption']
        answer = gpt_map_action(action)
        characters_to_remove = string.whitespace + string.punctuation
        answer = answer.strip(characters_to_remove)
        predicate = update_action_set(action_list, answer)
        extract_result.append({'id': id, 'action': action, 'predicate': predicate})
        with open(save_path, 'w') as f:
            json.dump(extract_result, f)
    return


def bddx_prepare(annotation_path, Video_folder, map_save_path, detect_save_path, vector_save_path, llm_prediction_path, llm_predicate_path):
    from pgm.BDDX_extractor import YOLO_detector
    query_annotation_csv(annotation_path, map_save_path)
    train_dict = json.load(open(map_save_path))
    yolo_dec = YOLO_detector(train_dict, Video_folder)
    yolo_dec.extract_classes(detect_save_path, annotation_path)
    map_LLM_pred(llm_prediction_path, llm_predicate_path)
    json_to_vectors(detect_save_path, vector_save_path, llm_predicate_path)
    return


def train_pipeline(train_data_path, config, weight_save_path):
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    train_data = np.array(data)
    pgm = PGM(config, learning_rate=1e-5, regularization=1e-5, max_iter=10000)
    weight = pgm.train_mln(train_data, weight_save_path)
    return weight 


def test_pipeline(test_data_path, weight_save_path):
    with open(test_data_path, 'rb') as f:
        data = pickle.load(f)
    test_data = np.array(data)
    pgm = PGM(weight_path=weight_save_path)
    accuracy = pgm.eval(test_data)
    return accuracy


def fake_data_generate(balance_num, truth_data_path):
    def violates_formulas(args, formulas):
        return any(formula(args) for formula in formulas)
    
    action_map = {
        'Keep': 'KEEP',
        'Accelerate': 'ACCELERATE',
        'Decelerate': 'DECELERATE',
        'Stop': 'STOP',
        'Reverse': 'REVERSE',
        'MakeLeftTurn': 'MAKE_LEFT_TURN',
        'MakeRightTurn': 'MAKE_RIGHT_TURN',
        'MakeUTurn': 'MAKE_U_TURN',
        'Merge': 'MERGE',
        'LeftPass': 'LEFT_PASS',
        'RightPass': 'RIGHT_PASS',
        'Yield': 'YIELD',
        'ChangeToLeftLane': 'CHANGE_TO_LEFT_LANE',
        'ChangeToRightLane': 'CHANGE_TO_RIGHT_LANE',
        'Park': 'PARK',
        'PullOver': 'PULL_OVER'
    }
    
    truth_data = json.load(open(truth_data_path))
    action_count = {}
    for item in truth_data:
        actions = item['predicate']
        for action in actions:
            if action in action_count:
                action_count[action] += 1
            else:
                action_count[action] = 1
    predicate_set = BDDX().predicate
    formula_set = BDDX().formulas
    fake_data=[]
    for action, count in action_count.items():
        if count < balance_num:
            fakedata = [0] * (BDDX().action_num+BDDX().condition_num)
            action_index = predicate_set[action_map[action]]
            fakedata[action_index] = 1
            
            valid_environment_predicates = []
            for _ in range(random.randint(0, 2)):
                temp_data = fakedata.copy()
                # Randomly select an environment predicate
                env_pred = random.choice(list(predicate_set.values())[16:28])
                temp_data[env_pred] = 1
                
                # Check if adding this predicate violates any formulas
                if not violates_formulas(temp_data, formula_set):
                    valid_environment_predicates.append(env_pred)
            
            # Update fakedata with valid environment predicates
            for pred in valid_environment_predicates:
                fakedata[pred] = 1
                
            if action in BDDX().velocityCS_list:
                fakedata[28 + ['KEEP', 'ACCELERATE', 'DECELERATE', 'STOP', 'REVERSE'].index(action)] = 1
            else:
                random_velocity = random.choice(['KEEP', 'ACCELERATE', 'DECELERATE', 'STOP', 'REVERSE'])
                fakedata[28 + ['KEEP', 'ACCELERATE', 'DECELERATE', 'STOP', 'REVERSE'].index(random_velocity)] = 1
                
            if action in ['MakeLeftTurn', 'LeftPass', 'ChangeToLeftLane']:
                fakedata[34] = 1  # LEFT_CS
            elif action in ['MakeRightTurn', 'RightPass', 'ChangeToRightLane']:
                fakedata[35] = 1  # RIGHT_CS
            else:
                random_direction = random.choice([33, 34, 35])  # LEFT_CS or RIGHT_CS
                fakedata[random_direction] = 1
                
            for i in range(36, 52):
                if random.random() < 0.9:
                    fakedata[i] = 1 if fakedata[i - 36] else 0
                else:
                    MLLM_predicate = random.choice([36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
                    fakedata[MLLM_predicate] = 1
            
            fake_data.append(fakedata)
    return fake_data
                
    
if __name__ == "__main__":
    
    map_LLM_pred('Data/BDDX/video_process/v9_top2.json', 'result/v9_top2/LLM_result.json')
    
    # data_path = 'result/ragdriver_kl_0.01-0.35_geo_unskew_filled_rag_top2_v8_train/LLM_result.json'
    # possible_worlds_count(data_path)
    # print(fake_data_generate(10, truth_data_path))