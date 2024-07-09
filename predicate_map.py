# predicate vector:

# Keep, Accelerate, Decelerate, Stop, MakeLeftTurn, MakeRightTurn, Merge, ChangeToLeftLane, ChangeToRightLane, PullOver
# TrafficLight, SolidGreenLight, SolidRedLight, SolidYellowLight, YellowLeftArrowLight, GreenLeftArrowLight, 
# RedLeftArrowLight, IntersectionAhead, MergingTrafficSign, WrongWaySign, KeepRightSign, LaneEndsSign, 
# NoLeftTurnSign, NoRightTurnSign, PedestrianCrossingSign, StopSign, ThruTrafficMergeLeftSign, RedYieldSign

# action_extract_map = {
#     ("stop",): "Stop",
#     ("forward",): "Keep",
#     ("accelerate",): "Accelerate",
#     ("brake",): "Decelerate",
#     ("merge",): "Merge",
#     ("right", "turn"): "MakeRightTurn",
#     ("right", "lane"): "ChangeToRightLane",
#     ("left", "turn"): "MakeLeftTurn",
#     ("left", "lane"): "ChangeToLeftLane",
#     ("pull", "over"): "PullOver",
#     ("continue",): "Keep",
# }

import json
import pickle

predicate_num = 28

action_map = {
    ("stop",): 3,
    ("stops",): 3,
    ("stopped",): 3,
    ("forward",): 0,
    ("accelerate",): 1,
    ("accelerates",): 1,
    ("accelerating",): 1,
    ("slow",): 2,
    ("slows",): 2,
    ("brake",): 2,
    ("merge",): 6,
    ("right", "turn"): 5,
    ("right", "lane"): 8,
    ("left", "turn"): 4,
    ("left", "lane"): 7,
    ("pull", "over"): 9,
    ("continue",): 0,
}

class_map = {
    "addedLane": 18,
    "curveLeft": None,
    "curveRight": None,
    "dip": None,
    "doNotEnter": 19,
    "doNotPass": None,
    "go": 11,
    "go forward": 11,
    "go forward traffic light": 11,
    "go left": 15,
    "go left traffic light": 15,
    "go traffic light": 11,
    "intersection": 17,
    "keepRight": 20,
    "laneEnds": 21,
    "merge": 18,
    "noLeftTurn": 22,
    "noRightTurn": 23,
    "pedestrianCrossing": 24,
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
    "stop": 25,
    "stop left": 16, 
    "stop left traffic light": 16, 
    "stop traffic light": 12, 
    "stopAhead": 25,
    "thruMergeLeft": 26,
    "thruMergeRight": None,
    "thruTrafficMergeLeft": 26,
    "truckSpeedLimit55": None,
    "turnLeft": None,
    "turnRight": None,
    "warning": 13, 
    "warning left": 14, 
    "warning left traffic light": 14, 
    "warning traffic light": 13, 
    "yield": 27,
    "yieldAhead": 27,
    "zoneAhead25": None,
    "zoneAhead45": None
}
    

def map_action_to_vector(action):
    
    vector = [0] * predicate_num
    
    input_str_lower = action.lower()
    words = input_str_lower.split()
    for i in range(len(words)):
        for keys, index in action_map.items():
            if all(key in words[i:i+len(keys)] for key in keys):
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
    if any(combined_vector[11:17]):
        combined_vector[10] = 1 # TrafficLight
    return combined_vector

def segment_to_vector(segment):
    action_vector = map_action_to_vector(segment["action"])
    class_vector = map_classes_to_vector(segment["classes"])
    return combine_vectors(action_vector, class_vector)

def json_to_vectors(data, train_data_savePth):
    vectors = []
    for video, segments in data.items():
        for segment_name, segment in segments.items():
            vector = segment_to_vector(segment)
            vectors.append(vector)
    with open(train_data_savePth, "wb") as f:
        pickle.dump(vectors, f)
        
    return
