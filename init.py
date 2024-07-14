import json
import pickle
import numpy as np
# from YOLO_detector import YOLO_detector
from video_annotation import query_annotation_csv
from predicate_map import json_to_vectors
from PGM import PGM


optimal_weight_path = 'optimal_weights.pkl'


def train_data_prepare(annotation_path, Video_folder, map_save_path, YOLO_detect_path, train_data_path, train_num):
    # query_annotation_csv(annotation_path, train_num, map_save_path)
    train_dict = json.load(open(map_save_path))
    # yolo_dec = YOLO_detector(train_dict, Video_folder)
    # yolo_dec.extract_classes(YOLO_detect_path)
    # json_to_vectors(YOLO_detect_path, train_data_path)
    return

def train_pipeline(train_data_path, validate_data_path, weight_save_path):
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    train_data = np.array(data)
    with open(validate_data_path, 'rb') as f:
        data = pickle.load(f)
    validate_data = np.array(data)
    pgm = PGM(learning_rate=1e-5, max_iter=10000)
    weight = pgm.train_mln(train_data, validate_data)
    np.save(weight_save_path, weight)
    print("Optimal weights saved to {}".format(weight_save_path))
    return 


def test_pipeline(test_data_path, weight_save_path):
    with open(test_data_path, 'rb') as f:
        data = pickle.load(f)
        
    test_data = np.array(data)
    pgm = PGM(weight_path=weight_save_path, learning_rate=1e-2, max_iter=10000)
    accuracy, precision, recall, f1_score = pgm.eval(test_data)
    return
    


def main():
    pattern = 'train'
    annotation_path = "Data/video_process/conversation_bddx_{}.json".format(pattern)
    Video_folder = "Data/BDD-X/Videos/videos/"
    map_save_path = 'map_ann_{}.json'.format(pattern)
    YOLO_detect_path = '{}_detected_classes.json'.format(pattern)
    train_data_path = 'train_vectors.pkl'
    test_data_path = 'test_vectors.pkl'
    train_num = -1
    
    weight_save_path = 'optimal_weights.npy'
    
    # train_data_prepare(annotation_path, Video_folder, map_save_path, YOLO_detect_path, train_data_path, train_num)
    train_pipeline(train_data_path, test_data_path, weight_save_path)
    
    # test_pipeline(test_data_path, weight_save_path)

if __name__ == "__main__":
    main()
    
    