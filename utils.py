import re
import os
import cv2
import json
import torch
import string
import pickle
import numpy as np
from tqdm import tqdm
# from pgm.PGM import PGM
from pgm.PGM_drivelm import PGM
from openai import OpenAI
from pgm.predicate_map import json_to_vectors
from pgm.video_annotation import query_annotation_csv
from transformers import AutoModelForCausalLM, AutoTokenizer

def Llama3_map_action(action):
    """
    map action to predicates
    """  
    model_path = "/home/xuanyang/data/Meta-Llama-3-8B-Instruct/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda:3")
    prompt = """Given the current behavior of the car, use one predicate to best describe the behavior of the car. If multiple actions are described, take the last one. The predicates are: 
    Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass, Yield, ChangeToLeftLane, ChangeToRightLane, Park, PullOver.
    Here are some examples:
    #Current Behavior#: The car is travelling down the road.
    #Predicates#: Keep\n
    #Current Behavior#: The car is making left turn.
    #Predicates#: MakeLeftTurn\n
    #Current Behavior#: The car is slowing down and then comes to a stop.
    #Predicates#: Stop\n
    #Current Behavior#: The car is making a left turn an then accelerates.
    #Predicates#: Accelerate\n
    #Current Behavior#: {action}
    #Predicates#: """.format(action=action)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:3")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
    generated_token = outputs[0][len(inputs['input_ids'][0]):]
    output_text = tokenizer.decode(generated_token, skip_special_tokens=True).strip()
    return output_text


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

    
def action_predicate_count():
    train_data = json.load(open('Data/video_process/new_conversation_bddx_train.json'))
    action_list = ['Keep', 'Accelerate', 'Decelerate', 'Stop', 'Reverse', 'MakeLeftTurn', 'MakeRightTurn', 
                   'MakeUTurn', 'Merge', 'LeftPass', 'RightPass', 'Yield', 'ChangeToLeftLane', 
                   'ChangeToRightLane', 'Park', 'PullOver']
    
    map_action_count = {}
    
    for item in tqdm(train_data):
        action = item['conversations'][1]['value']
        output = gpt_map_action(action, isPred=False)
        map_action = update_action(action_list, output)
        if map_action in ['MakeUTurn','Yield','LeftPass:','RightPass','Park','Reverse','PullOver']:
            print(f"{action}: {map_action}")
        if map_action in map_action_count:
            map_action_count[map_action] += 1
        else:
            map_action_count[map_action] = 1
    for action, count in map_action_count.items():
        print(f"{action}: {count}")
    
    with open('map_action_count.json', 'w') as f:
        json.dump(map_action_count, f)
    return map_action_count


def pkl_reader(npy_file):
    with open(npy_file, 'rb') as f:
        data = pickle.load(f)
    return data


def action_counter(json_path):
    data = json.load(open(json_path))
    action_count = {}
    for item in data:
        action = item['action']
        if action in action_count:
            action_count[action] += 1
        else:
            action_count[action] = 1
    return action_count


def video_snapshot(video_path, output_folder, start_second, end_second, interval=1):
    video_name = video_path.split('/')[-1].split('.')[0] + str(start_second) + '_' + str(end_second)
    output_path = os.path.join(output_folder, video_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cap = cv2.VideoCapture(video_path)    
    fps = cap.get(cv2.CAP_PROP_FPS)    
    start_frame = start_second * fps
    end_frame = end_second * fps
    frame_count = 0
    image_count = start_second
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # read video from start_frame to end_frame, and save picture every 1 seconds
        if frame_count >= start_frame and frame_count <= end_frame:
            if frame_count % round(interval * fps) == 0:
                image_name = os.path.join(output_path, f'{image_count}.jpg')
                cv2.imwrite(image_name, frame)
                image_count += 1
            
        frame_count += 1
    cap.release()
    return


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
    action_list=['Keep', 'Accelerate', 'Decelerate', 'Stop', 'Reverse', 
                'MakeLeftTurn', 'MakeRightTurn', 'MakeUTurn', 'Merge', 
                'LeftPass', 'RightPass', 'Yield', 'ChangeToLeftLane',
                'ChangeToRightLane', 'Park', 'PullOver']
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


def data_prepare(annotation_path, Video_folder, map_save_path, YOLO_detect_path, vector_data_path, segment_num):
    from pgm.YOLO_detector import YOLO_detector
    # query_annotation_csv(annotation_path, segment_num, map_save_path)
    train_dict = json.load(open(map_save_path))
    yolo_dec = YOLO_detector(train_dict, Video_folder)
    yolo_dec.extract_classes(YOLO_detect_path)
    json_to_vectors(YOLO_detect_path, vector_data_path)
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