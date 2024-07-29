import os
import cv2
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
from pgm.PGM import PGM
from pgm.predicate_map import json_to_vectors
from pgm.video_annotation import query_annotation_csv
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/xuanyang/data/Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda:2")

def Llama3_map_action(action):
    
    prompt_1 = "The current behavior of the car: "
    prompt_2 = "Which of the following actions most closely represents the current behavior of the car:\n"
    prompt_3 = "Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass, Yield, ChangeToLeftLane, ChangeToRightLane, ChangeToCenterLeftTurnLane, Park, PullOver.\n"
    # prompt_4 = "You must and can only choose one, and your answer needs to contain only your answer, without adding other explanations or extraneous content.\n"
    prompt_4 = "You can choose as many actions as you want, as long as it's closest to the original behavior\n"
    prompt_5 = "Answer:"
    input_text = prompt_1 + action + "\n" + prompt_2 + prompt_3 + prompt_4 + prompt_5
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda:2")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=4)
    generated_token = outputs[0][len(inputs['input_ids'][0]):]
    output_text = tokenizer.decode(generated_token, skip_special_tokens=True).strip()
    return output_text


def update_action(action_list, action):
    map_actions = set()
    for act in action_list:
        if act.lower() in action.lower():
            map_actions.add(act)
    return frozenset(map_actions)


def action_predicate_count():
    train_data = json.load(open('Data/video_process/conversation_bddx_train.json'))
    action_list = ['Keep', 'Accelerate', 'Decelerate', 'Stop', 'Reverse', 'MakeLeftTurn', 'MakeRightTurn', 
                   'MakeUTurn', 'Merge', 'LeftPass', 'RightPass', 'Yield', 'ChangeToLeftLane', 
                   'ChangeToRightLane', 'ChangeToCenterLeftTurnLane', 'Park', 'PullOver']
    
    map_action_count = {}
    
    for item in tqdm(train_data):
        action = item['conversations'][1]['value']
        output = Llama3_map_action(action)
        map_actions = update_action(action_list, output)
        if map_actions in map_action_count:
            map_action_count[map_actions] += 1
        else:
            map_action_count[map_actions] = 1
    
    for actions, count in map_action_count.items():
        print(f"{set(actions)}: {count}")
        
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


def data_prepare(annotation_path, Video_folder, map_save_path, YOLO_detect_path, vector_data_path, segment_num):
    from pgm.YOLO_detector import YOLO_detector
    # query_annotation_csv(annotation_path, segment_num, map_save_path)
    train_dict = json.load(open(map_save_path))
    yolo_dec = YOLO_detector(train_dict, Video_folder)
    yolo_dec.extract_classes(YOLO_detect_path)
    json_to_vectors(YOLO_detect_path, vector_data_path)
    return

def train_pipeline(train_data_path, validate_data_path, weight_save_path):
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    train_data = np.array(data)
    with open(validate_data_path, 'rb') as f:
        data = pickle.load(f)
    validate_data = np.array(data)
    pgm = PGM(learning_rate=1e-5, regularization=1e-5, max_iter=10000)
    weight = pgm.train_mln(train_data, weight_save_path, validate_data)
    return 


def test_pipeline(test_data_path, weight_save_path):
    with open(test_data_path, 'rb') as f:
        data = pickle.load(f)
        
    test_data = np.array(data)
    pgm = PGM(weight_path=weight_save_path)
    accuracy = pgm.eval(test_data)
    return accuracy



if __name__ == "__main__":
    map_action_count = action_predicate_count()
    with open('action_predicates_count.json', 'w') as f:
        json.dump(map_action_count, f)
        
    

