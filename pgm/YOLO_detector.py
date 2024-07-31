import cv2
import json
import torch
import string
import logging
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
from utils import Llama3_map_action
logging.getLogger('ultralytics').setLevel(logging.ERROR)


def detect_single_frame(data_dict):
    yolo = YOLO('best.pt')
    video_path = data_dict['original_video']
    try:
        start_time = float(data_dict['start_time'])
        end_time = float(data_dict['end_time'])
    except ValueError:
        print("Invalid start_time or end_time:{}".format(video_path))
        return []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {video_path}")
    
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    if start_frame >= total_frames or end_frame > total_frames or start_frame >= end_frame:
        print("Invalid start_time or end_time")
        cap.release()
        return []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read the end frame")
        cap.release()
        return []
    
    yolo_results = set()
    detections = yolo(frame)
    detected_classes = []
    for detection in detections:
        for box in detection.boxes:
            class_name = yolo.names[int(box.cls)]
            detected_classes.append(class_name)
    if detected_classes:
            yolo_results.update(detected_classes)
    
    cap.release()
    return list(yolo_results)


class YOLO_detector:
    
    def __init__(self, video_dict, Video_folder):
        self.model = YOLO('best.pt')
        self.dict = video_dict
        self.Video_folder = Video_folder
    
    def load_video(self, video_index):
        video_path = self.Video_folder + video_index
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")
        return cap
    
    def detect_objects_yolo(self, frame):
        results = self.model(frame)
        detections = []
        for result in results:
            for box in result.boxes:
                class_name = self.model.names[int(box.cls)]
                detections.append(class_name)
        return detections
    
    # detection whole segment
    def get_yolo_results_for_segment(self, video_index, start_time, end_time):
        
        try:
            start_time = float(start_time)
            end_time = float(end_time)
        except ValueError:
            print("Invalid start_time or end_time:{}".format(video_index))
            return []
            
        cap = self.load_video(video_index)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the frame range for the specified time window
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)
        if start_frame >= total_frames or end_frame > total_frames or start_frame >= end_frame:
            print("Invalid start_time or end_time")
            cap.release()
            return []

        yolo_results = set()
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= start_frame:
                if frame_idx > end_frame:
                    break
                detections = self.detect_objects_yolo(frame)
                if detections:
                    yolo_results.update(detections)
                
            frame_idx += 1
        
        cap.release()
        return list(yolo_results)
    
    
    # detect from last frame
    def get_yolo_results_for_last_frame(self, video_index, start_time, end_time):
        try:
            start_time = float(start_time)
            end_time = float(end_time)
        except ValueError:
            print("Invalid start_time or end_time:{}".format(video_index))
            return []
            
        cap = self.load_video(video_index)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the frame range for the specified time window
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)
        if start_frame >= total_frames or end_frame > total_frames or start_frame > end_frame:
            print("Invalid start_time or end_time, start:{}, end:{}, total:{}".format(start_frame, end_frame, total_frames))
            cap.release()
            return []

        uni_samples = 8
        step = max((end_frame - start_frame) // (uni_samples - 1), 1)
        
        sample_last_frame = start_frame + step * (uni_samples - 1)
        
        sample_end_frame = min(sample_last_frame, end_frame, total_frames - 1)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample_end_frame)
        ret, frame = cap.read()
        if not ret:
            print("Failed to read the end frame, total:{}, sample_end:{}".format(total_frames, sample_end_frame))
            cap.release()
            return []

        yolo_results = set()
        detections = self.detect_objects_yolo(frame)
        if detections:
            yolo_results.update(detections)

        cap.release()
        return list(yolo_results)

    
    
    def extract_classes(self, save_path):
        
        extracted_data = []
        
        action_list=['Keep', 'Accelerate', 'Decelerate', 'Stop', 'Reverse', 
                     'MakeLeftTurn', 'MakeRightTurn', 'MakeUTurn', 'Merge', 
                     'LeftPass', 'RightPass']
        
        print('Extracting classes from YOLO...')
        
        for item in tqdm(self.dict):
            
            video_path = item['original_video']
            start_time = item['start_time']
            end_time = item['end_time']
            
            # yolo_results = self.get_yolo_results_for_segment(video_path, start_time, end_time)
            yolo_results = self.get_yolo_results_for_last_frame(video_path, start_time, end_time)
            
            if not yolo_results:
                continue
            
            answer = Llama3_map_action(item['action'])
            characters_to_remove = string.whitespace + string.punctuation
            answer = answer.strip(characters_to_remove)
            action = None
            for act in action_list:
                if act.lower() in answer.lower():
                    action = act
                    break
                
            if action is None:
                continue
            
            extracted_data.append({
                'id': item['id'],
                'video': video_path,
                'action': action,
                'classes': yolo_results
            })
        
        with open(save_path, 'w') as f:
            json.dump(extracted_data, f, indent=4)
  
        return extracted_data
    
    
