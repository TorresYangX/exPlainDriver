import json
import pickle
from tqdm import tqdm

# Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass,(10) 
# Yield, ChangeToLeftLane, ChangeToRightLane, ChangeToCenterLeftTurnLane, Park, PullOver(16)
# SolidRedLight, SolidYellowLight, YellowLeftArrowLight,(19)
# RedLeftArrowLight, MergingTrafficSign, WrongWaySign,(22)
# NoLeftTurnSign, NoRightTurnSign, PedestrianCrossingSign, StopSign, RedYieldSign, DoNotPassSign, SlowSign(29)

predicate_num = 36

action_map = {
    "Keep": 0,
    "Accelerate": 1,
    "Decelerate": 2,
    "Stop": 3,
    "Reverse": 4,
    "MakeLeftTurn": 5,
    "MakeRightTurn": 6,
    "MakeUTurn": 7,
    "Merge": 8,
    "LeftPass": 9,
    "RightPass": 10,
    "Yield": 11,
    "ChangeToLeftLane": 12,
    "ChangeToRightLane": 13,
    "Park": 14,
    "PullOver": 15
}

class_map = {
    "stop traffic light": 16, 
    "warning": 17, 
    "warning traffic light": 17, 
    "warning left": 18, 
    "warning left traffic light": 18, 
    "stop left": 19, 
    "stop left traffic light": 19, 
    "merge": 20,
    "noLeftTurn": 21,
    "noRightTurn": 22,
    "pedestrianCrossing": 23,
    "stop": 24,
    "stopAhead": 24,
    "yield": 25,
    "yieldAhead": 25,
    "slow": 26,
    "go": 27,
    "go forward": 27,
    "go forward traffic light": 27,
    
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
    "Keep": 28,
    "Accelerate": 29,
    "Decelerate": 30,
    "Stop": 31,
    "Reverse": 32,
    "Straight": 33,
    "Left": 34,
    "Right": 35,
}
    

def map_action_to_vector(action):
    
    vector = [0] * predicate_num
    
    # if action is not None
    if action and action in action_map:
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


def combine_vectors(action_vector, class_vector, cs_vector):
    combined_vector = [max(a, c, cs) for a, c, cs in zip(action_vector, class_vector, cs_vector)]
    return combined_vector

def segment_to_vector(segment):
    action_vector = map_action_to_vector(segment["action"])
    class_vector = map_classes_to_vector(segment["classes"])
    cs_vector = map_cs_to_vector(segment["velocity_predicate"], segment["direction_predicate"])
    return combine_vectors(action_vector, class_vector, cs_vector)

def json_to_vectors(YOLO_detect_path, train_data_savePth):
    data = json.load(open(YOLO_detect_path))
    vectors = []
    for item in tqdm(data):
        vector = segment_to_vector(item)
        vectors.append(vector)
    with open(train_data_savePth, "wb") as f:
        pickle.dump(vectors, f)
        
    return
