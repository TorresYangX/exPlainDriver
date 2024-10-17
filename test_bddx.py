import os
import json
import numpy as np
import pandas as pd
from pgm.PGM import PGM
from tqdm import tqdm
from pgm.config import BDDX
from pgm.predicate_map import segment_to_vector
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def id2action(yolo_detect_Test, id):
    for item in yolo_detect_Test:
        if item['id'] == id:
            return item['action'], item['classes'], item['velocity_predicate'], item['direction_predicate']

def predicate_judge(action_list, ground_action_list):
    return all(elem in ground_action_list for elem in action_list)


def BDDX_Test(weights, detection_result, LLM_predicate_path):
    pgm = PGM(weights=weights, config=BDDX())
    yolo_detect_Test = json.load(open(detection_result)) 
    LLM_result_predicate = json.load(open(LLM_predicate_path))
    total = 0
    correct_pgm = 0
    correct = 0
    for item in tqdm(LLM_result_predicate):
        id = item['id']
        try:
            ground_action, yolo_class, v_p, d_p = id2action(yolo_detect_Test, id)
        except:
            ground_action = None
            continue
        total += 1
        for p_item in LLM_result_predicate:
            if p_item['id'] == id:
                ori_LLM_action = p_item['predicate']
                break
        if predicate_judge(ori_LLM_action, ground_action):
            correct += 1
        acc = correct / total
        LLM_action = []
        # judge action in action_list separately
        for action in ori_LLM_action:
            instance = {
                "action": [action],
                "classes": yolo_class,
                'velocity_predicate': v_p,
                'direction_predicate': d_p
            }    
            llm_prediction = [action]
            instance_vector = np.array(segment_to_vector(instance, llm_prediction))
            condition = instance_vector[BDDX().action_num:]
            _, index = pgm.infer_action_probability(condition)
            LLM_action.append([index])
        if predicate_judge(LLM_action, ground_action):
            correct_pgm += 1
    acc_pgm = correct_pgm / total
    logger.info(f"Total: {total}, PGM Accuracy: {acc_pgm}, Acc: {acc}")
    return 0


# if __name__ == "__main__":
#     yolo_detect_Test_path = 'Data/BDDX/process_data/eval_filled/eval_filled_detected_classes.json'
#     weights = np.load('weights/optimal_weights_bddx_filled_rag_top2_v8.npy')         
#     LLM_PGM_acc(weights, BDDX(), yolo_detect_Test_path)
    
        