import json
from YOLO_detector import YOLO_detector
from video_annotation import query_annotation_csv
from predicate_map import json_to_vectors
from PGM import train, infer

annotation_path = "Data/BDD-X-Dataset/BDD-X-Annotations_v1.csv"
Video_folder = 'Data/BDD-X/Videos/videos/'
YOLO_detect_path = 'detected_classes.json'
train_data_path = 'train_vectors.pkl'
optimal_weight_path = 'optimal_weights.pkl'

if __name__=="__main__":
    video_dict = query_annotation_csv(annotation_path)
    yolo_detector = YOLO_detector(video_dict, Video_folder)
    yolo_detector.extract_classes_per_segment(YOLO_detect_path)
    data = json.load(open(YOLO_detect_path))
    json_to_vectors(data, train_data_path)
    train(train_data_path, optimal_weight_path)
    infer(optimal_weight_path, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    