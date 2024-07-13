import json
from YOLO_detector import YOLO_detector
from video_annotation import query_annotation_csv
from predicate_map import json_to_vectors
from PGM import train, infer

annotation_path = "Data/video_process/conversation_bddx_train.json"
Video_folder = "Data/video_process/BDDX_Processed/"
YOLO_detect_path = 'detected_classes.json'
train_data_path = 'train_vectors.pkl'
optimal_weight_path = 'optimal_weights.pkl'

train_num = 3

if __name__=="__main__":
    train_dict = query_annotation_csv(annotation_path, train_num)
    yolo_dec = YOLO_detector(train_dict, Video_folder)
    yolo_dec.extract_classes(YOLO_detect_path)
    json_to_vectors(YOLO_detect_path, train_data_path)
    train(train_data_path, optimal_weight_path)
    infer(optimal_weight_path, [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    