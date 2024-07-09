import cv2
import numpy as np
from ultralytics import YOLO
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
import json


class YOLO_detector:
    
    def __init__(self, video_dict, Video_folder):
        self.model = YOLO('best.pt')
        self.dict = video_dict
        self.video_indices = {video: idx for idx, video in enumerate(video_dict)}
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
    
    def get_yolo_results_for_video(self, video_index):
        cap = self.load_video(video_index)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = total_frames / frame_rate
        print(f"Video length: {video_length:.2f}s, Frame rate: {frame_rate:.2f}fps, Total frames: {total_frames}")

        yolo_results = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            time_in_seconds = frame_idx / frame_rate
            detections = self.detect_objects_yolo(frame)
            if detections:
                for detection in detections:
                    yolo_results.append({'time': time_in_seconds, 'class': detection})
            else:
                yolo_results.append({'time': time_in_seconds, 'class': None})
                
            frame_idx += 1
        
        cap.release()
        return yolo_results
    
    def extract_classes_per_segment(self, save_path):
        
        extracted_classes = {}
        
        for video_index in self.video_indices:
            yolo_results = self.get_yolo_results_for_video(video_index)
        
            segments_classes = {}
            
            i = 1
            while f'Answer.{i}start' in self.dict[video_index]:
                start_time = float(self.dict[video_index][f'Answer.{i}start'])
                end_time = float(self.dict[video_index][f'Answer.{i}end'])
                action = self.dict[video_index][f'Answer.{i}action']
                
                segment_key = f'Segment {i}'
                segments_classes[segment_key] = {
                    'action': action,
                    'classes': set()
                }
                
                for detection in yolo_results:
                    detection_time = detection['time']
                    if start_time <= detection_time <= end_time:
                        segments_classes[segment_key]['classes'].add(detection['class'])
                
                i += 1
            
            for segment_key in segments_classes:
                segments_classes[segment_key]['classes'] = list(segments_classes[segment_key]['classes'])
            
            extracted_classes[video_index] = segments_classes
        
        with open(save_path, 'w') as f:
            json.dump(extracted_classes, f, indent=4)
  
        return 0