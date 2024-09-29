import json
import pickle
from tqdm import tqdm

predicate_num = 43
action_num = 7
llm_action_num = 7


action_map = {
    "Noraml": 0,
    "Fast": 1,
    "Slow": 2,
    "Stop": 3,
    "Left": 4,
    "Right": 5,
    "Straight": 6,
}

class_map = {
    "stop traffic light": 7, 
    "warning": 8, 
    "warning traffic light": 8, 
    "warning left": 9, 
    "warning left traffic light": 9, 
    "stop left": 10, 
    "stop left traffic light": 10, 
    "merge": 11,
    "noLeftTurn": 12,
    "noRightTurn": 13,
    "pedestrianCrossing": 14,
    "stop": 15,
    "stopAhead": 15,
    "yield": 16,
    "yieldAhead": 16,
    "slow": 17,
    "go": 18,
    "go forward": 18,
    "go forward traffic light": 18,
    'DOUBLE_DASHED_WHITE_LEFT': 19,
    'DOUBLE_DASHED_WHITE_RIGHT': 20,
    'SINGLE_SOLID_WHITE_LEFT': 21,
    'SINGLE_SOLID_WHITE_RIGHT': 22,
    'DOUBLE_SOLID_WHITE_LEFT': 23,
    'DOUBLE_SOLID_WHITE_RIGHT': 24,
    'SINGLE_ZIGZAG_WHITE_LEFT': 25,
    'SINGLE_ZIGZAG_WHITE_RIGHT': 26,
    'SINGLE_SOLID_YELLOW_LEFT': 27,
    'SINGLE_SOLID_YELLOW_RIGHT': 28,
    
    
    "intersection": None,
    "addedLane": None,
    "keepRight": None,
    "laneEnds": None,
    "thruMergeLeft": None,
    "thruTrafficMergeLeft": None,
    "go left": None,
    "go left traffic light": None,
    "curveLeft": None,
    "curveRight": None,
    "dip": None,
    "rampSpeedAdvisory20": None,
    "rampSpeedAdvisory35": None,
    "rampSpeedAdvisory40": None,
    "rampSpeedAdvisory45": None,
    "rampSpeedAdvisory50": None,
    "rampSpeedAdvisoryUrdbl": None,
    "rightLaneMustTurn": None,
    "roundabout": None,
    "school": None,
    "schoolSpeedLimit25": None,
    "signalAhead": None,
    "speedLimit15": None,
    "speedLimit25": None,
    "speedLimit30": None,
    "speedLimit35": None,
    "speedLimit40": None,
    "speedLimit45": None,
    "speedLimit50": None,
    "speedLimit55": None,
    "speedLimit65": None,
    "speedLimitUrdbl": None,
    "thruMergeRight": None,
    "truckSpeedLimit55": None,
    "turnLeft": None,
    "turnRight": None,
    "zoneAhead25": None,
    "zoneAhead45": None
}

cs_map = {
    'Normal': 29,
    'Fast': 30,
    'Slow': 31,
    'Stop': 32,
    'Left': 33,
    'Right': 34,
    'Straight': 35,
}

LLM_action_map = {
    'Normal': 36,
    'Fast': 37,
    'Slow': 38,
    'Stop': 39,
    'Left': 40,
    'Right': 41,
    'Straight': 42,
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

def id2prediction(id, prediction_data):
    for item in prediction_data:
        if item["image_id"] == id:
            return item["action"]

def json_to_vectors(YOLO_detect_path, train_data_savePth, llm_prediction_path):
    print('begin to convert json to vectors...')
    data = json.load(open(YOLO_detect_path))
    
    conv_data = json.load(open('Data/DriveLM/DriveLM_process/conversation_drivelm_train.json', 'r'))
    conv_data_ids = [item['id'] for item in conv_data]
    
    vectors = []
    prediction_data = json.load(open(llm_prediction_path))
    for item in tqdm(data):
        id = item["image_id"]
        if id not in conv_data_ids:
            continue
        llm_prediction = id2prediction(id, prediction_data)
        llm_prediction = item["action"]
        vector = segment_to_vector(item, llm_prediction)
        vectors.append(vector)
    print(len(vectors), len(vectors[0]))
    with open(train_data_savePth, "wb") as f:
        pickle.dump(vectors, f)  
    return

def combine_condition_vectors(class_vector, cs_vector):
    combined_vector = [max(c, cs) for c, cs in zip(class_vector, cs_vector)]
    return combined_vector

def segment_to_condition_vector(segment):
    class_vector = map_classes_to_vector(segment["classes"])
    cs_vector = map_cs_to_vector(segment["velocity_predicate"], segment["direction_predicate"])
    return combine_condition_vectors(class_vector, cs_vector)


def json_to_condition_vectors(YOLO_detect_path, train_data_savePth):
    print('begin to convert json to condition vectors...')
    data = json.load(open(YOLO_detect_path))
    
    conv_data = json.load(open('Data/DriveLM/DriveLM_process/conversation_drivelm_test.json', 'r'))
    conv_data_ids = [item['id'] for item in conv_data]
    
    vectors = []
    for item in tqdm(data):
        id = item["image_id"]
        if id not in conv_data_ids:
            continue
        vector = segment_to_condition_vector(item)[action_num: predicate_num-llm_action_num]
        vectors.append(vector)
    print(len(vectors), len(vectors[0]))
    with open(train_data_savePth, "wb") as f:
        pickle.dump(vectors, f)  
    return