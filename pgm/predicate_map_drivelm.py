import json
import pickle
from tqdm import tqdm

predicate_num = 32

action_map = {
    "Keep": 0,
    "Accelerate": 1,
    "Decelerate": 2,
    "Stop": 3,
    "MakeLeftTurn": 4,
    "MakeRightTurn": 5,
    "ChangeToLeftLane": 6,
    "ChangeToRightLane": 7,
    "Straight": 8,
}

class_map = {
    "stop traffic light": 9, 
    "warning": 10, 
    "warning traffic light": 10, 
    "warning left": 11, 
    "warning left traffic light": 11, 
    "stop left": 12, 
    "stop left traffic light": 12, 
    "merge": 13,
    "noLeftTurn": 14,
    "noRightTurn": 15,
    "pedestrianCrossing": 16,
    "stop": 17,
    "stopAhead": 17,
    "yield": 18,
    "yieldAhead": 18,
    "slow": 19,
    "go": 20,
    "go forward": 20,
    "go forward traffic light": 20,
    
    'stopLine': 21,
    'DOUBLE_DASHED_WHITE_LEFT': 22,
    'DOUBLE_DASHED_WHITE_RIGHT': 23,
    'SINGLE_SOLID_WHITE_LEFT': 24,
    'SINGLE_SOLID_WHITE_RIGHT': 25,
    'DOUBLE_SOLID_WHITE_LEFT': 26,
    'DOUBLE_SOLID_WHITE_RIGHT': 27,
    'SINGLE_ZIGZAG_WHITE_LEFT': 28,
    'SINGLE_ZIGZAG_WHITE_RIGHT': 29,
    'SINGLE_SOLID_YELLOW_LEFT': 30,
    'SINGLE_SOLID_YELLOW_RIGHT': 31,
    
    
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

def combine_vectors(action_vector, class_vector):
    combined_vector = [max(a, c) for a, c in zip(action_vector, class_vector)]
    return combined_vector


def segment_to_vector(segment):
    action_vector = map_action_to_vector(segment["action"])
    class_vector = map_classes_to_vector(segment["classes"])
    return combine_vectors(action_vector, class_vector)


def json_to_vectors(YOLO_detect_path, train_data_savePth):
    data = json.load(open(YOLO_detect_path))
    vectors = []
    for item in tqdm(data):
        vector = segment_to_vector(item)
        vectors.append(vector)
    print(len(vectors), len(vectors[0]))
    with open(train_data_savePth, "wb") as f:
        pickle.dump(vectors, f)  
    return

if __name__ == "__main__":
    YOLO_detect_path = "process_data_drivelm/test/test_detected_classes.json"
    train_data_savePth = "process_data_drivelm/test/test_vectors.pkl"
    json_to_vectors(YOLO_detect_path, train_data_savePth)
    
    # condition_vector_path = "process_data_drivelm/test/test_condition_vectors.pkl"
    # train_vectors = pickle.load(open(train_data_savePth, "rb"))
    # print(len(train_vectors), len(train_vectors[0]))
    # condition_vectors = [row[8:] for row in train_vectors]
    # print(len(condition_vectors), len(condition_vectors[0]))
    # with open(condition_vector_path, "wb") as f:
    #     pickle.dump(condition_vectors, f)
    