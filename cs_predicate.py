import re
import os
from openai import OpenAI
import json
from tqdm import tqdm
def control_signal_extractor(cs_string):
    pattern = r"(\w+): \[(.*?)\]"
    matches = re.findall(pattern, cs_string)
    control_signals = {match[0]: eval(f"[{match[1]}]") for match in matches}
    return control_signals

def gpt_map_cs(Speed, Course):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    system_prompt = "You are a helpful assistant"
    prompt = """
    Given the current speed and course of the car, use one velocity predicate and one directional predicate to best describe the behavior of the car. 
    The velocity predicates are: Normal, Fast, Slow, Stop.
    The directional predicates are: Straight, Left, Right. 
    Output the predicates directly without any additional information.
    Here are some examples:
    # Speed: [(4.54, 0.0), (5.34, 0.0), (5.67, 0.0), (5.7, 0.0), (6.46, 0.0), (6.63, 0.0)]
    # Course: [(1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)]
    # Predicate: Fast, Straight
    # Speed: [(10.01, 0.0), (9.88, 0.0), (9.52, 0.0), (9.39, 0.0), (9.15, 0.0), (8.94, 0.0)]
    # Course: [(0.84, 0.0), (0.84, 0.0), (0.86, 0.0), (0.89, 0.0), (0.93, 0.0), (0.95, 0.0)]
    # Predicate: Fast, Right
    # Speed: [(2.51, 0.0), (2.49, 0.0), (2.45, 0.0), (2.43, 0.0), (2.43, 0.0), (2.37, 0.0)]
    # Course: [(0.85, 0.0), (0.85, 0.0), (0.86, 0.0), (0.85, 0.0), (0.82, 0.0), (0.75, 0.0)]
    # Predicate: Slowly, Left
    # Speed: [(1.65, 0.0), (1.37, 0.0), (0.73, 0.0), (0.09, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]
    # Course: [(0.86, 0.0), (0.86, 0.0), (0.87, 0.0), (0.86, 0.0), (0.86, 0.0), (0.86, 0.0), (0.85, 0.0), (0.84, 0.0)]
    # Predicate: Stop, Straight
    # Speed: {speed}
    # Course: {course}
    # Predicate: """.format(speed=Speed, course=Course)
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

if __name__ == '__main__':
    # origin_path = 'process_data_drivelm/train/train_detected_classes.json'
    # conv_path = 'DriveLM_process/conversation_drivelm_train.json'
    # ori_data = json.load(open(origin_path, 'r'))
    # conv_data = json.load(open(conv_path, 'r'))
    
    # new_items = []
    # for item in tqdm(ori_data):
    #     id = item['image_id']
    #     action_list = item['action']
    #     classes = item['classes']
    #     action_mapping = {
    #         'Keep': 'Normal',
    #         'Accelerate': 'Fast',
    #         'Decelerate': 'Slow',
    #         'Stop': 'Stop',
    #         'ChangeToLeftLane': 'Straight',
    #         'ChangeToRightLane': 'Straight',
    #         'MakeLeftTurn': 'Left',
    #         'MakeRightTurn': 'Right'
    #     }
    #     new_action_list = []
    #     for action in action_list:
    #         if action in action_mapping.keys():
    #             new_action_list.append(action_mapping[action])
    #         else:
    #             new_action_list.append(action)
    #     for conv in conv_data:
    #         if conv['id'] == id:
    #             cs_string = conv['conversations'][-2]['value']
    #             cs_info = control_signal_extractor(cs_string)
    #             break
    #     answer = gpt_map_cs(cs_info['Speed'], cs_info['Orientation'])
    #     velocity_predicate = update_action(['Normal', 'Fast', 'Slow', 'Stop'], answer)
    #     direction_predicate = update_action(['Straight', 'Left', 'Right'], answer)
    #     if velocity_predicate is None or direction_predicate is None:
    #         print('error')
    #     new_items.append({
    #         'image_id': id,
    #         'classes': classes,
    #         'action': new_action_list,
    #         'velocity_predicate': velocity_predicate,
    #         'direction_predicate': direction_predicate
    #     })
    #     with open('process_data_drivelm/train/train_detected_classes_with_predicate.json', 'w') as f:
    #         json.dump(new_items, f)
    
    action_mapping = {
        'Keep': 'Normal',
        'Accelerate': 'Fast',
        'Decelerate': 'Slow',
        'Stop': 'Stop',
        'ChangeToLeftLane': 'Straight',
        'ChangeToRightLane': 'Straight',
        'MakeLeftTurn': 'Left',
        'MakeRightTurn': 'Right'
    }
    origin_path = 'result/drivelm/LLM_result_predicate.json'
    ori_data = json.load(open(origin_path, 'r'))
    new_items = []
    for item in tqdm(ori_data):
        id = item['image_id']
        option = item['option']
        action_list = item['action_list']
        new_action_list = []
        for action in action_list:
            if action in action_mapping.keys():
                new_action_list.append(action_mapping[action])
            else:
                new_action_list.append(action)
        new_items.append({
            'image_id': id,
            'option': option,
            'action': new_action_list
        })
        with open('result/drivelm/LLM_result_predicate_with_action.json', 'w') as f:
            json.dump(new_items, f)