import json
import pickle
from tqdm import tqdm
from config import BDDX
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predicate_num = BDDX().action_num+BDDX().condition_num
action_map = {
    "Keep": BDDX().predicate["KEEP"],
    "Accelerate": BDDX().predicate["ACCELERATE"],
    "Decelerate": BDDX().predicate["DECELERATE"],
    "Stop": BDDX().predicate["STOP"],
    "Reverse": BDDX().predicate["REVERSE"],
    "MakeLeftTurn": BDDX().predicate["MAKE_LEFT_TURN"],
    "MakeRightTurn": BDDX().predicate["MAKE_RIGHT_TURN"],
    "MakeUTurn": BDDX().predicate["MAKE_U_TURN"],
    "Merge": BDDX().predicate["MERGE"],
    "LeftPass": BDDX().predicate["LEFT_PASS"],
    "RightPass": BDDX().predicate["RIGHT_PASS"],
    "Yield": BDDX().predicate["YIELD"],
    "ChangeToLeftLane": BDDX().predicate["CHANGE_TO_LEFT_LANE"],
    "ChangeToRightLane": BDDX().predicate["CHANGE_TO_RIGHT_LANE"],
    "Park": BDDX().predicate["PARK"],
    "PullOver": BDDX().predicate["PULL_OVER"]
}
class_map = {
    "stop traffic light": BDDX().predicate["SOLID_RED_LIGHT"], 
    "warning": BDDX().predicate["SOLID_YELLOW_LIGHT"], 
    "warning traffic light": BDDX().predicate["SOLID_YELLOW_LIGHT"], 
    "warning left": BDDX().predicate["YELLOW_LEFT_ARROW_LIGHT"], 
    "warning left traffic light": BDDX().predicate["YELLOW_LEFT_ARROW_LIGHT"], 
    "stop left": BDDX().predicate["RED_LEFT_ARROW_LIGHT"], 
    "stop left traffic light": BDDX().predicate["RED_LEFT_ARROW_LIGHT"], 
    "merge": BDDX().predicate["MERGING_TRAFFIC_SIGN"],
    "noLeftTurn": BDDX().predicate["NO_LEFT_TURN_SIGN"],
    "noRightTurn": BDDX().predicate["NO_RIGHT_TURN_SIGN"],
    "pedestrianCrossing": BDDX().predicate["PEDESTRIAN_CROSSING_SIGN"],
    "stop": BDDX().predicate["STOP_SIGN"],
    "stopAhead": BDDX().predicate["STOP_SIGN"],
    "yield": BDDX().predicate["RED_YIELD_SIGN"],
    "yieldAhead": BDDX().predicate["RED_YIELD_SIGN"],
    "slow": BDDX().predicate["SLOW_SIGN"],
    "go": BDDX().predicate["SOLID_GREEN_LIGHT"],
    "go forward": BDDX().predicate["SOLID_GREEN_LIGHT"],
    "go forward traffic light": BDDX().predicate["SOLID_GREEN_LIGHT"],
}
cs_map = {
    "Keep": BDDX().predicate["KEEP_CS"],
    "Accelerate": BDDX().predicate["ACCELERATE_CS"],
    "Decelerate": BDDX().predicate["DECELERATE_CS"],
    "Stop": BDDX().predicate["STOP_CS"],
    "Reverse": BDDX().predicate["REVERSE_CS"],
    "Straight": BDDX().predicate["STRAIGHT_CS"],
    "Left": BDDX().predicate["LEFT_CS"],
    "Right": BDDX().predicate["RIGHT_CS"],
}
LLM_action_map = {
    "Keep": BDDX().predicate["KEEP_LLM"],
    "Accelerate": BDDX().predicate["ACCELERATE_LLM"],
    "Decelerate": BDDX().predicate["DECELERATE_LLM"],
    "Stop": BDDX().predicate["STOP_LLM"],
    "Reverse": BDDX().predicate["REVERSE_LLM"],
    "MakeLeftTurn": BDDX().predicate["MAKE_LEFT_TURN_LLM"],
    "MakeRightTurn": BDDX().predicate["MAKE_RIGHT_TURN_LLM"],
    "MakeUTurn": BDDX().predicate["MAKE_U_TURN_LLM"],
    "Merge": BDDX().predicate["MERGE_LLM"],
    "LeftPass": BDDX().predicate["LEFT_PASS_LLM"],
    "RightPass": BDDX().predicate["RIGHT_PASS_LLM"],
    "Yield": BDDX().predicate["YIELD_LLM"],
    "ChangeToLeftLane": BDDX().predicate["CHANGE_TO_LEFT_LANE_LLM"],
    "ChangeToRightLane": BDDX().predicate["CHANGE_TO_RIGHT_LANE_LLM"],
    "Park": BDDX().predicate["PARK_LLM"],
    "PullOver": BDDX().predicate["PULL_OVER_LLM"]
}
    

def map_action_to_vector(action_list):
    vector = [0] * predicate_num
    # if action is not None
    if action_list:
        for action in action_list:
            if action in action_map:
                index = action_map[action]
                vector[index] = 1
    return vector

def map_classes_to_vector(classes):
    vector = [0] * predicate_num
    if classes:
        for cls in classes:
            if cls and cls in class_map:
                index = class_map[cls]
                if index is not None:
                    vector[index] = 1
    return vector

def map_cs_to_vector(velocity, direction):
    vector = [0] * predicate_num
    if velocity and direction:
        vector[cs_map[direction]] = 1
        vector[cs_map[velocity]] = 1
    return vector

def map_llm_to_vector(action_list):
    vector = [0] * predicate_num
    if action_list:
        for action in action_list:
            if action in LLM_action_map:
                index = LLM_action_map[action]
                vector[index] = 1
    return vector


def combine_vectors(action_vector, class_vector, cs_vector, llm_vector):
    combined_vector = [max(a, c, cs, llm) for a, c, cs, llm in zip(action_vector, class_vector, cs_vector, llm_vector)]
    return combined_vector

def segment_to_vector(segment, llm_prediction):
    action_vector = map_action_to_vector(segment["action"])
    class_vector = map_classes_to_vector(segment["classes"])
    cs_vector = map_cs_to_vector(segment["velocity_predicate"], segment["direction_predicate"])
    llm_vector = map_llm_to_vector(llm_prediction)
    return combine_vectors(action_vector, class_vector, cs_vector, llm_vector)

def id2prediction(id, prediction_path):
    prediction_data = json.load(open(prediction_path))
    for item in prediction_data:
        if item["id"] == id:
            return item["predicate"]

def json_to_vectors(YOLO_detect_path, data_savePth, llm_prediction_path):
    logger.info('Begin to convert json to vectors...')
    data = json.load(open(YOLO_detect_path))
    vectors = []
    for item in tqdm(data):
        id = item["id"]
        llm_prediction = id2prediction(id, llm_prediction_path)
        vector = segment_to_vector(item, llm_prediction)
        vectors.append(vector)
    logger.info('vectors shape:', len(vectors), len(vectors[0]))
    with open(data_savePth, "wb") as f:
        pickle.dump(vectors, f)  
    return
