# predicate vector:

# Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass, Yield
# ChangeToLeftLane, ChangeToRightLane, ChangeToCenterLeftTurnLane, Park, PullOver(16)
# TrafficLight, SolidGreenLight, SolidRedLight, SolidYellowLight, YellowLeftArrowLight, GreenLeftArrowLight, (22)
# RedLeftArrowLight, IntersectionAhead, MergingTrafficSign, WrongWaySign, KeepRightSign, LaneEndsSign, (28)
# NoLeftTurnSign, NoRightTurnSign, PedestrianCrossingSign, StopSign, ThruTrafficMergeLeftSign, RedYieldSign, DoNotPassSign, SlowSign

import json
import pickle
from tqdm import tqdm

# action_map = {
#     "Keep": 0, 
#     "Accelerate": 1, 
#     "Decelerate": 2, 
#     "Stop": 3, 
#     "Reverse": 4, 
#     "MakeLeftTurn": 5, 
#     "MakeRightTurn": 6, 
#     "MakeUTurn": 7, 
#     "Merge": 8, 
#     "LeftPass": 9, 
#     "RightPass": 10, 
#     "Yield": 11,
#     "ChangeToLeftLane": 12, 
#     "ChangeToRightLane": 13, 
#     "ChangeToCenterLeftTurnLane": 14, 
#     "Park": 15, 
#     "PullOver": 16
# }

# class_map = {
#     "go": 18,
#     "go forward": 18,
#     "go forward traffic light": 18,
#     "go left": 22,
#     "go left traffic light": 22,
#     "go traffic light": 18,
#     "warning": 20, 
#     "warning left": 21, 
#     "warning left traffic light": 21, 
#     "warning traffic light": 20, 
#     "stop": 19,
#     "stop left": 23, 
#     "stop left traffic light": 23, 
#     "stop traffic light": 19, 
#     "intersection": 24,
#     "addedLane": 25,
#     "merge": 25,
#     "doNotEnter": 26,
#     "keepRight": 27,
#     "laneEnds": 28,
#     "noLeftTurn": 29,
#     "noRightTurn": 30,
#     "pedestrianCrossing": 31,
#     "stopAhead": 32,
#     "thruMergeLeft": 33,
#     "thruTrafficMergeLeft": 33,
#     "yield": 34,
#     "yieldAhead": 34,
#     "doNotPass": 35,
#     "slow": 36,
#     "curveLeft": None,
#     "curveRight": None,
#     "dip": None,
#     "rampSpeedAdvisory20": None,
#     "rampSpeedAdvisory35": None,
#     "rampSpeedAdvisory40": None,
#     "rampSpeedAdvisory45": None,
#     "rampSpeedAdvisory50": None,
#     "rampSpeedAdvisoryUrdbl": None,
#     "rightLaneMustTurn": None,
#     "roundabout": None,
#     "school": None,
#     "schoolSpeedLimit25": None,
#     "signalAhead": None,
#     "speedLimit15": None,
#     "speedLimit25": None,
#     "speedLimit30": None,
#     "speedLimit35": None,
#     "speedLimit40": None,
#     "speedLimit45": None,
#     "speedLimit50": None,
#     "speedLimit55": None,
#     "speedLimit65": None,
#     "speedLimitUrdbl": None,
#     "thruMergeRight": None,
#     "truckSpeedLimit55": None,
#     "turnLeft": None,
#     "turnRight": None,
#     "zoneAhead25": None,
#     "zoneAhead45": None
# }


# Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, LeftPass, RightPass (7)
# SolidRedLight, SolidYellowLight, YellowLeftArrowLight,(11)
# RedLeftArrowLight, MergingTrafficSign, WrongWaySign,(14)
# NoLeftTurnSign, NoRightTurnSign, PedestrianCrossingSign, StopSign, RedYieldSign, DoNotPassSign, SlowSign(21)

predicate_num = 21

action_map = {
    "Decelerate": 0, 
    "Stop": 1, 
    "Reverse": 2, 
    "MakeLeftTurn": 3, 
    "MakeRightTurn": 4, 
    "MakeUTurn": 5, 
    "LeftPass": 6, 
    "RightPass": 7, 
}

class_map = {
    "stop": 8,
    "stop traffic light": 8, 
    "warning": 9, 
    "warning traffic light": 9, 
    "warning left": 10, 
    "warning left traffic light": 10, 
    "stop left": 11, 
    "stop left traffic light": 11, 
    "merge": 12,
    "doNotEnter": 13,
    "noLeftTurn": 14,
    "noRightTurn": 15,
    "pedestrianCrossing": 16,
    "stopAhead": 17,
    "yield": 18,
    "yieldAhead": 18,
    "doNotPass": 19,
    "slow": 20,
    
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
