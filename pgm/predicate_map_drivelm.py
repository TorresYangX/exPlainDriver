import json
import pickle
from tqdm import tqdm
from pgm.config import DriveLM
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predicate_num = DriveLM().action_num+DriveLM().condition_num
action_num = DriveLM().action_num
llm_action_num = DriveLM().action_num


action_map = {
    "Noraml": DriveLM().predicate["NORMAL"],
    "Fast": DriveLM().predicate["FAST"],
    "Slow": DriveLM().predicate["SLOW"],
    "Stop": DriveLM().predicate["STOP"],
    "Left": DriveLM().predicate["LEFT"],
    "Right": DriveLM().predicate["RIGHT"],
    "Straight": DriveLM().predicate["STRAIGHT"],
}
class_map = {
    "stop traffic light": DriveLM().predicate["SOLID_RED_LIGHT"], 
    "warning": DriveLM().predicate["SOLID_YELLOW_LIGHT"], 
    "warning traffic light": DriveLM().predicate["SOLID_YELLOW_LIGHT"], 
    "warning left": DriveLM().predicate["YELLOW_LEFT_ARROW_LIGHT"], 
    "warning left traffic light": DriveLM().predicate["YELLOW_LEFT_ARROW_LIGHT"], 
    "stop left": DriveLM().predicate["RED_LEFT_ARROW_LIGHT"], 
    "stop left traffic light": DriveLM().predicate["RED_LEFT_ARROW_LIGHT"], 
    "merge": DriveLM().predicate["MERGING_TRAFFIC_SIGN"],
    "noLeftTurn": DriveLM().predicate["NO_LEFT_TURN_SIGN"],
    "noRightTurn": DriveLM().predicate["NO_RIGHT_TURN_SIGN"],
    "pedestrianCrossing": DriveLM().predicate["PEDESTRIAN_CROSSING_SIGN"],
    "stop": DriveLM().predicate["STOP_SIGN"],
    "stopAhead": DriveLM().predicate["STOP_SIGN"],
    "yield": DriveLM().predicate["RED_YIELD_SIGN"],
    "yieldAhead": DriveLM().predicate["RED_YIELD_SIGN"],
    "slow": DriveLM().predicate["SLOW_SIGN"],
    "go": DriveLM().predicate["SOLID_GREEN_LIGHT"],
    "go forward": DriveLM().predicate["SOLID_GREEN_LIGHT"],
    "go forward traffic light": DriveLM().predicate["SOLID_GREEN_LIGHT"],
    'DOUBLE_DASHED_WHITE_LEFT': DriveLM().predicate['DOUBLE_DASHED_WHITE_LINE_LEFT'],
    'DOUBLE_DASHED_WHITE_RIGHT': DriveLM().predicate['DOUBLE_DASHED_WHITE_LINE_RIGHT'],
    'SINGLE_SOLID_WHITE_LEFT': DriveLM().predicate['SINGLE_SOLID_WHITE_LINE_LEFT'],
    'SINGLE_SOLID_WHITE_RIGHT': DriveLM().predicate['SINGLE_SOLID_WHITE_LINE_RIGHT'],
    'DOUBLE_SOLID_WHITE_LEFT': DriveLM().predicate['DOUBLE_SOLID_WHITE_LINE_LEFT'],
    'DOUBLE_SOLID_WHITE_RIGHT': DriveLM().predicate['DOUBLE_SOLID_WHITE_LINE_RIGHT'],
    'SINGLE_ZIGZAG_WHITE_LEFT': DriveLM().predicate['SINGLE_ZIGZAG_WHITE_LINE_LEFT'],
    'SINGLE_ZIGZAG_WHITE_RIGHT': DriveLM().predicate['SINGLE_ZIGZAG_WHITE_LINE_RIGHT'],
    'SINGLE_SOLID_YELLOW_LEFT': DriveLM().predicate['SINGLE_SOLID_YELLOW_LINE_LEFT'],
    'SINGLE_SOLID_YELLOW_RIGHT': DriveLM().predicate['SINGLE_SOLID_YELLOW_LINE_RIGHT'],
}
cs_map = {
    'Normal': DriveLM().predicate['NORMAL_CS'],
    'Fast': DriveLM().predicate['FAST_CS'],
    'Slow': DriveLM().predicate['SLOW_CS'],
    'Stop': DriveLM().predicate['STOP_CS'],
    'Left': DriveLM().predicate['LEFT_CS'],
    'Right': DriveLM().predicate['RIGHT_CS'],
    'Straight': DriveLM().predicate['STRAIGHT_CS'],
}
LLM_action_map = {
    'Normal': DriveLM().predicate['NORMAL_LLM'],
    'Fast': DriveLM().predicate['FAST_LLM'],
    'Slow': DriveLM().predicate['SLOW_LLM'],
    'Stop': DriveLM().predicate['STOP_LLM'],
    'Left': DriveLM().predicate['LEFT_LLM'],
    'Right': DriveLM().predicate['RIGHT_LLM'],
    'Straight': DriveLM().predicate['STRAIGHT_LLM'],
}


def map_action_to_vector(action_list):
    vector = [0] * predicate_num
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

def json_to_vectors(detect_save_path, train_data_savePth, llm_predicate_path):
    logger.info('begin to convert json to vectors...')
    data = json.load(open(detect_save_path))    
    vectors = []
    prediction_data = json.load(open(llm_predicate_path))
    for item in tqdm(data):
        llm_prediction = id2prediction(id, prediction_data)
        llm_prediction = item["action"]
        vector = segment_to_vector(item, llm_prediction)
        vectors.append(vector)
    logger.info('vectors shape:', len(vectors), len(vectors[0]))
    with open(train_data_savePth, "wb") as f:
        pickle.dump(vectors, f)  
    return

# def combine_condition_vectors(class_vector, cs_vector):
#     combined_vector = [max(c, cs) for c, cs in zip(class_vector, cs_vector)]
#     return combined_vector

# def segment_to_condition_vector(segment):
#     class_vector = map_classes_to_vector(segment["classes"])
#     cs_vector = map_cs_to_vector(segment["velocity_predicate"], segment["direction_predicate"])
#     return combine_condition_vectors(class_vector, cs_vector)


# def json_to_condition_vectors(YOLO_detect_path, train_data_savePth):
#     print('begin to convert json to condition vectors...')
#     data = json.load(open(YOLO_detect_path))
    
#     conv_data = json.load(open('Data/DriveLM/DriveLM_process/conversation_drivelm_test.json', 'r'))
#     conv_data_ids = [item['id'] for item in conv_data]
    
#     vectors = []
#     for item in tqdm(data):
#         id = item["image_id"]
#         if id not in conv_data_ids:
#             continue
#         vector = segment_to_condition_vector(item)[action_num: predicate_num-llm_action_num]
#         vectors.append(vector)
#     print(len(vectors), len(vectors[0]))
#     with open(train_data_savePth, "wb") as f:
#         pickle.dump(vectors, f)  
#     return