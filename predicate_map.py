# predicate vector:

# Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass, Yield
# ChangeToLeftLane, ChangeToRightLane, ChangeToCenterLeftTurnLane, Park, PullOver
# TrafficLight, SolidGreenLight, SolidRedLight, SolidYellowLight, YellowLeftArrowLight, GreenLeftArrowLight, 
# RedLeftArrowLight, IntersectionAhead, MergingTrafficSign, WrongWaySign, KeepRightSign, LaneEndsSign, 
# NoLeftTurnSign, NoRightTurnSign, PedestrianCrossingSign, StopSign, ThruTrafficMergeLeftSign, RedYieldSign

import json
import pickle

predicate_num = 35

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
    "addedLane": 25,
    "curveLeft": None,
    "curveRight": None,
    "dip": None,
    "doNotEnter": 26,
    "doNotPass": None,
    "go": 18,
    "go forward": 18,
    "go forward traffic light": 18,
    "go left": 22,
    "go left traffic light": 22,
    "go traffic light": 18,
    "intersection": 24,
    "keepRight": 27,
    "laneEnds": 28,
    "merge": 25,
    "noLeftTurn": 29,
    "noRightTurn": 30,
    "pedestrianCrossing": 31,
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
    "slow": None,
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
    "stop": 32,
    "stop left": 23, 
    "stop left traffic light": 23, 
    "stop traffic light": 19, 
    "stopAhead": 32,
    "thruMergeLeft": 33,
    "thruMergeRight": None,
    "thruTrafficMergeLeft": 33,
    "truckSpeedLimit55": None,
    "turnLeft": None,
    "turnRight": None,
    "warning": 20, 
    "warning left": 21, 
    "warning left traffic light": 21, 
    "warning traffic light": 20, 
    "yield": 34,
    "yieldAhead": 34,
    "zoneAhead25": None,
    "zoneAhead45": None
}
    

def map_action_to_vector(action):
    
    vector = [0] * predicate_num
    
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
    if any(combined_vector[18:24]):
        combined_vector[17] = 1 # TrafficLight
    return combined_vector

def segment_to_vector(segment):
    action_vector = map_action_to_vector(segment["action"])
    class_vector = map_classes_to_vector(segment["classes"])
    return combine_vectors(action_vector, class_vector)

def json_to_vectors(YOLO_detect_path, train_data_savePth):
    
    data = json.load(open(YOLO_detect_path))
    
    vectors = []
    for item in data:
        vector = segment_to_vector(item)
        vectors.append(vector)
    with open(train_data_savePth, "wb") as f:
        pickle.dump(vectors, f)
        
    return
