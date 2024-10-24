import json
import re
from tqdm import tqdm
from openai import OpenAI
from pgm.predicate_map_drivelm import json_to_vectors
import os
from pgm.DriveLM_extractor import DriveLM_extractor
import pickle
# from pgm.PGM_drivelm import PGM
from arixv.MLN_sole.MLN import PGM
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        r'steering to the left': 'Left',
        r'steering to the right': 'Right',
    }
    actions = []
    matched_patterns = set()
    for pattern,action in mapping_rules.items():
        if re.search(pattern, sentence, re.IGNORECASE):
            if 'steering' in pattern:
                if 'slightly' in pattern or 'steering' not in matched_patterns:
                    matched_patterns.add('steering')
                    actions.append(action)
            else:
                actions.append(action)
    return actions

def get_option(text, option_letter):
    pattern = rf"{option_letter}\.\s(.+?)(?=\s[A-Z]\.|$)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
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
        try:
            action_list = action_map(option)
            if not action_list or len(action_list) != 2:
                print(f"Error: {scene_id}, {frame_id}, {option}")
            info = {
                "image_id": pred_list[0]['image_id'],
                "option": option,
                "action": action_list
            }
            infos.append(info)
            with open(info_save_path, 'w') as f:
                json.dump(infos, f, indent=4)
        except:
            print(f"Error: {scene_id}, {frame_id}, {option}")
            continue
    logger.info(f"Total {len(infos)} action predicates extracted.")
    

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


def drivelm_prepare(conv_path, question_path, detect_save_path, vector_data_path, llm_prediction_path, llm_predicate_path):
    drive_extractor = DriveLM_extractor()
    drive_extractor.condition_predicate_extractor(conv_path, question_path, detect_save_path)
    pred_action_predicate_extractor(llm_prediction_path, question_path, llm_prediction_path)
    json_to_vectors(detect_save_path, vector_data_path, llm_predicate_path)
    return


def train_pipeline(train_data_path, config, weight_save_path):
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    train_data = np.array(data)
    pgm = PGM(config, learning_rate=1e-5, regularization=1e-5, max_iter=10000)
    weight = pgm.train_mln(train_data, weight_save_path)
    return weight 
            

        
                    