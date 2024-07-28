import json
import pickle
from tqdm import tqdm

# Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass,(10) 
# Yield, ChangeToLeftLane, ChangeToRightLane, ChangeToCenterLeftTurnLane, Park, PullOver(16)
# SolidRedLight, SolidYellowLight, YellowLeftArrowLight,(19)
# RedLeftArrowLight, MergingTrafficSign, WrongWaySign,(22)
# NoLeftTurnSign, NoRightTurnSign, PedestrianCrossingSign, StopSign, RedYieldSign, DoNotPassSign, SlowSign(29)

predicate_num = 30

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
    "ChangeToCenterLeftTurnLane": 14,
    "Park": 15,
    "PullOver": 16
}

class_map = {
    "stop traffic light": 17, 
    "warning": 18, 
    "warning traffic light": 18, 
    "warning left": 19, 
    "warning left traffic light": 19, 
    "stop left": 20, 
    "stop left traffic light": 20, 
    "merge": 21,
    "doNotEnter": 22,
    "noLeftTurn": 23,
    "noRightTurn": 24,
    "pedestrianCrossing": 25,
    "stop": 26,
    "stopAhead": 26,
    "yield": 27,
    "yieldAhead": 27,
    "doNotPass": 28,
    "slow": 29,
    
    "intersection": None,
    "addedLane": None,
    "keepRight": None,
    "laneEnds": None,
    "thruMergeLeft": None,
    "thruTrafficMergeLeft": None,
    "go": None,
    "go forward": None,
    "go forward traffic light": None,
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
    with open(train_data_savePth, "wb") as f:
        pickle.dump(vectors, f)
        
    return
