import pickle
import json
import cv2
import os
import numpy as np
from pgm.YOLO_detector import YOLO_detector
from pgm.video_annotation import query_annotation_csv
from pgm.predicate_map import json_to_vectors
from pgm.PGM import PGM

def pkl_reader(npy_file):
    with open(npy_file, 'rb') as f:
        data = pickle.load(f)
    return data

def action_counter(json_path):
    data = json.load(open(json_path))
    action_count = {}
    for item in data:
        action = item['action']
        if action in action_count:
            action_count[action] += 1
        else:
            action_count[action] = 1
    return action_count


def video_snapshot(video_path, output_folder, start_second, end_second, interval=1):
    video_name = video_path.split('/')[-1].split('.')[0] + str(start_second) + '_' + str(end_second)
    output_path = os.path.join(output_folder, video_name)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
           
    cap = cv2.VideoCapture(video_path)    
    fps = cap.get(cv2.CAP_PROP_FPS)    
    start_frame = start_second * fps
    end_frame = end_second * fps
        
    frame_count = 0
    image_count = start_second
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # read video from start_frame to end_frame, and save picture every 1 seconds
        if frame_count >= start_frame and frame_count <= end_frame:
            if frame_count % round(interval * fps) == 0:
                image_name = os.path.join(output_path, f'{image_count}.jpg')
                cv2.imwrite(image_name, frame)
                image_count += 1
            
        frame_count += 1
        
    cap.release()
    return


def data_prepare(annotation_path, Video_folder, map_save_path, YOLO_detect_path, vector_data_path, segment_num):
    query_annotation_csv(annotation_path, segment_num, map_save_path)
    train_dict = json.load(open(map_save_path))
    yolo_dec = YOLO_detector(train_dict, Video_folder)
    yolo_dec.extract_classes(YOLO_detect_path)
    json_to_vectors(YOLO_detect_path, vector_data_path)
    return

def train_pipeline(train_data_path, validate_data_path, weight_save_path):
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    train_data = np.array(data)
    with open(validate_data_path, 'rb') as f:
        data = pickle.load(f)
    validate_data = np.array(data)
    pgm = PGM(learning_rate=1e-5, regularization=1e-5, max_iter=10000)
    weight = pgm.train_mln(train_data, weight_save_path, validate_data)
    return 


def test_pipeline(test_data_path, weight_save_path):
    with open(test_data_path, 'rb') as f:
        data = pickle.load(f)
        
    test_data = np.array(data)
    pgm = PGM(weight_path=weight_save_path)
    accuracy = pgm.eval(test_data)
    return accuracy

def inference_pipeline(test_data, weight_save_path):
    pgm = PGM(weight_path=weight_save_path)
    prob, action_index = pgm.infer_action_probability(test_data)
    return prob, action_index

