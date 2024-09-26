import json
import pickle
from tqdm import tqdm

predicate_num = 36

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
    "stop left": 8, 
    "stop left traffic light": 8, 
    "noLeftTurn": 9,
    "noRightTurn": 10,
    "slow": 11,
    'DOUBLE_DASHED_WHITE_LEFT': 12,
    'DOUBLE_DASHED_WHITE_RIGHT': 13,
    'SINGLE_SOLID_WHITE_LEFT': 14,
    'SINGLE_SOLID_WHITE_RIGHT': 15,
    'DOUBLE_SOLID_WHITE_LEFT': 16,
    'DOUBLE_SOLID_WHITE_RIGHT': 17,
    'SINGLE_ZIGZAG_WHITE_LEFT': 18,
    'SINGLE_ZIGZAG_WHITE_RIGHT': 19,
    'SINGLE_SOLID_YELLOW_LEFT': 20,
    'SINGLE_SOLID_YELLOW_RIGHT': 21,
    
    
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
    'Normal': 22,
    'Fast': 23,
    'Slow': 24,
    'Stop': 25,
    'Left': 26,
    'Right': 27,
    'Straight': 28,
}

LLM_action_map = {
    'Normal': 29,
    'Fast': 30,
    'Slow': 31,
    'Stop': 32,
    'Left': 33,
    'Right': 34,
    'Straight': 35,
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
    vectors = []
    prediction_data = json.load(open(llm_prediction_path))
    for item in tqdm(data):
        id = item["image_id"]
        llm_prediction = id2prediction(id, prediction_data)
        llm_prediction = item["action"]
        vector = segment_to_vector(item, llm_prediction)
        vectors.append(vector)
    print(len(vectors), len(vectors[0]))
    with open(train_data_savePth, "wb") as f:
        pickle.dump(vectors, f)  
    return

# def combine_vectors(action_vector, class_vector, cs_vector):
#     combined_vector = [max(a, c, cs) for a, c, cs in zip(action_vector, class_vector, cs_vector)]
#     return combined_vector

# def segment_to_vector(segment):
#     action_vector = map_action_to_vector(segment["action"])
#     class_vector = map_classes_to_vector(segment["classes"])
#     cs_vector = map_cs_to_vector(segment["velocity_predicate"], segment["direction_predicate"])
#     return combine_vectors(action_vector, class_vector, cs_vector)


# def json_to_vectors(YOLO_detect_path, train_data_savePth):
#     print('begin to convert json to vectors...')
#     data = json.load(open(YOLO_detect_path))
    
#     conv_data = json.load(open('Data/DriveLM/DriveLM_process/conversation_drivelm_train.json', 'r'))
    
#     conv_data_ids = [item['id'] for item in conv_data]
    
#     vectors = []
#     for item in tqdm(data):
#         id = item["image_id"]
#         if id not in conv_data_ids:
#             continue
#         vector = segment_to_vector(item)
#         vectors.append(vector)
#     print(len(vectors), len(vectors[0]))
#     with open(train_data_savePth, "wb") as f:
#         pickle.dump(vectors, f)  
#     return