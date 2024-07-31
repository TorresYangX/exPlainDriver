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
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda:3")

def Llama3_map_action(action):
    
    """
    map action to predicates
    """
    
    prompt = """Given the current behavior of the car, use one predicate or the combination of two predicates to best describe the behavior of the car. The predicates are: 
    Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass, Yield, ChangeToLeftLane, ChangeToRightLane, Park, PullOver.
    Here are some examples:
    #Current Behavior#: The car is travelling down the road.
    #Predicates#: Keep\n
    #Current Behavior#: The car is moving forward down the highway.
    #Predicates#: Keep\n
    #Current Behavior#: The car is slowing down and then comes to a stop.
    #Predicates#: Decelerate, Stop\n
    #Current Behavior#: The car is making a left turn an then accelerates.
    #Predicates#: MakeLeftTurn, Accelerate\n
    #Current Behavior#: {action}
    #Predicates#: """.format(action=action)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:3")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)
    generated_token = outputs[0][len(inputs['input_ids'][0]):]
    output_text = tokenizer.decode(generated_token, skip_special_tokens=True).strip()
    return output_text


def update_action(action_list, action):
    map_actions = set()
    for act in action_list:
        if act.lower() in action.lower():
            map_actions.add(act)
    return frozenset(map_actions)


def LLM_compare_action(ground_action, pred_action):
    """
    directly compare the ground truth action and the predicted action
    """
    
    prompt = '''Given the question and its corresponding ground answer, determine whether the response from the LLM aligns with the ground answer. Respond with 'Yes' if the response matches the ground answer or 'No' if it is incorrect.(If the response contains all ground answer and have more infomation, consider it correct). Here are examples:
    #Question#: What is the action of ego car?
    #Ground Answer#: The car is moving forward down the road.
    #LLM Response#: The car moves forward then comes to a stop.
    #Answer#: Yes\n
    #Question#: What is the action of ego car?
    #Ground Answer#: The car is travelling forward.
    #LLM Response#: The car is driving forward with its windshield wipers on.
    #Answer#: Yes\n
    #Question#: What is the action of ego car?
    #Ground Answer#: The car changes lanes.
    #LLM Response#: The car is driving forward with its windshield wipers on.
    #Answer#: No\n
    #Question#: What is the action of ego car?
    #Ground Answer#: {answer}
    #LLM Response#: {response}
    #Answer#:'''.format(answer=ground_action, response=pred_action)
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:3")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1)
    generated_token = outputs[0][len(inputs['input_ids'][0]):]
    output_text = tokenizer.decode(generated_token, skip_special_tokens=True).strip()
    return output_text
    
    


def action_predicate_count():
    train_data = json.load(open('Data/video_process/conversation_bddx_train.json'))
    action_list = ['Keep', 'Accelerate', 'Decelerate', 'Stop', 'Reverse', 'MakeLeftTurn', 'MakeRightTurn', 
                   'MakeUTurn', 'Merge', 'LeftPass', 'RightPass', 'Yield', 'ChangeToLeftLane', 
                   'ChangeToRightLane', 'Park', 'PullOver']
    
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
    return weight 


def test_pipeline(test_data_path, weight_save_path):
    with open(test_data_path, 'rb') as f:
        data = pickle.load(f)
        
    test_data = np.array(data)
    pgm = PGM(weight_path=weight_save_path)
    accuracy = pgm.eval(test_data)
    return accuracy


if __name__ == "__main__":
    action = "The car is."
    answer = Llama3_map_action(action)
    print(answer)
    

