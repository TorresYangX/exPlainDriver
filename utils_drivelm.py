import json
import re
from tqdm import tqdm
from openai import OpenAI
import os


def action_map(sentence):
    mapping_rules = {
        r'going straight': 'Straight',
        r'driving fast': 'Fast',
        r'driving very fast': 'Fast',
        r'driving slowly': 'Slow',
        r'driving with normal speed': 'Normal',
        r'not moving': 'Stop',
        r'slightly steering to the left': 'Straight',
        r'slightly steering to the right': 'Straight',
        r'steering to the right': 'Left',
        r'steering to the left': 'Right',
    }
    actions = []
    matched_patterns = set()
    for pattern,action in mapping_rules.items():
        if re.search(pattern, sentence, re.IGNORECASE):
            if 'steering' in pattern:  # 处理 steering 行为，防止多次匹配
                if 'slightly' in pattern or 'steering' not in matched_patterns:
                    matched_patterns.add('steering')
                    actions.append(action)
            else:
                actions.append(action)
    return actions

def get_option(text, option_letter):
    pattern = rf"{option_letter}\.\s(.+?)(?=\s[A-Z]\.|$)"  # 匹配指定的选项，直到下一个选项或文本末尾
    match = re.search(pattern, text, re.DOTALL)  # 使用 DOTALL 使 . 可以匹配换行符
    if match:
        return match.group(1).strip()  # 返回匹配到的选项内容并去掉前后空格
    else:
        return None
    

def pred_action_predicate_extractor(pred_path, question_path, info_save_path):
    with open(pred_path, 'r') as f:
        preds = json.load(f)
    with open(question_path, 'r') as f:
        questions = json.load(f)
    infos = []
    for pred_list in tqdm(preds):
        scene_id = pred_list[0]['image_id'].split('_')[0]
        frame_id = pred_list[0]['image_id'].split('_')[1]
        question_part = questions[scene_id]["key_frames"][frame_id]["QA"]["behavior"][0]["Q"]
        pred_answer = pred_list[-2]['caption']
        option = get_option(question_part, pred_answer)
        action_list = action_map(option)
        if not action_list or len(action_list) != 2:
            print(f"Error: {scene_id}, {frame_id}, {option}")
        info = {
            "image_id": pred_list[0]['image_id'],
            "option": option,
            "action_list": action_list
        }
        infos.append(info)
        with open(info_save_path, 'w') as f:
            json.dump(infos, f, indent=4)
            

        
                    