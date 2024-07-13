import cv2
import numpy as np
from ultralytics import YOLO
import logging
logging.getLogger('ultralytics').setLevel(logging.ERROR)
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/xuanyang/data/Meta-Llama-3-8B-Instruct/"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda:2")

def Llama3_map(action):
    
    prompt_1 = "The current behavior of the car: \n"
    
    prompt_2 = "Which of the following actions most closely represents the current behavior of the car:\n"
    
    prompt_3 = "Keep, Accelerate, Decelerate, Stop, Reverse, MakeLeftTurn, MakeRightTurn, MakeUTurn, Merge, LeftPass, RightPass, Yield, ChangeToLeftLane, ChangeToRightLane, ChangeToCenterLeftTurnLane, Park, PullOver.\n"
    
    prompt_4 = "You must and can only choose one, and your answer needs to contain only your answer, without adding other explanations or extraneous content."
    
    input_text = prompt_1 + action + "\n" + prompt_2 + prompt_3 + prompt_4
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda:2")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=200)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output_text

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
    
    def get_yolo_results_for_video(self, video_index):
        cap = self.load_video(video_index)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_length = total_frames / frame_rate
        print(f"Video length: {video_length:.2f}s, Frame rate: {frame_rate:.2f}fps, Total frames: {total_frames}")

        yolo_results = set()
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.detect_objects_yolo(frame)
            if detections:
                yolo_results.update(detections)
                
            frame_idx += 1
        
        cap.release()
        return list(yolo_results)
    
    
    def extract_classes(self, save_path):
        
        extracted_data = []
        
        action_list=["Keep", "Accelerate", "Decelerate", "Stop", "Reverse",
        "MakeLeftTurn", "MakeRightTurn", "MakeUTurn", "Merge",
        "LeftPass", "RightPass", "Yield", "ChangeToLeftLane",
        "ChangeToRightLane", "ChangeToCenterLeftTurnLane",
        "Park", "PullOver"]
        
        for item in self.dict:
            
            video_path = item['video']
            
            yolo_results = self.get_yolo_results_for_video(video_path)
            print(item['action'])
            action_rough = Llama3_map(item['action'])
            print(action_rough)
            question_end = action_rough.rfind("\n\n")
            action = None
            if question_end != -1:
                answer = action_rough[question_end:].strip()
                for act in action_list:
                    if act in answer:
                        action = act
            
            extracted_data.append({
                'id': item['id'],
                'video': video_path,
                'action': action,
                'classes': yolo_results
            })
        
        with open(save_path, 'w') as f:
            json.dump(extracted_data, f, indent=4)
  
        return extracted_data