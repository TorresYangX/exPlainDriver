from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
from ultralytics import YOLO
import json
from tqdm import tqdm
import re

def action_map(sentence):
    mapping_rules = {
        r'going straight': 'Straight',
        r'driving fast': 'Fast',
        r'driving very fast': 'Fast',
        r'driving slowly': 'Slow',
        r'driving with normal speed': 'Normal',
        r'not moving': 'Stop',
        r'slightly steering to the left': 'Straight',
        r'slightly steering to the right': 'Straight',
        r'steering to the left': 'Left',
        r'steering to the right': 'Right',
    }
    actions = []
    matched_patterns = set()
    for pattern,action in mapping_rules.items():
        if re.search(pattern, sentence, re.IGNORECASE):
            if 'steering' in pattern:
                if 'slightly' in pattern or 'steering' not in matched_patterns:
                    matched_patterns.add('steering')
                    actions.append(action)
            else:
                actions.append(action)
    return actions

def get_option(text, option_letter):
    pattern = rf"{option_letter}\.\s(.+?)(?=\s[A-Z]\.|$)"
    match = re.search(pattern, text, re.DOTALL)  
    if match:
        return match.group(1).strip()
    else:
        return None


class DriveLM_extractor:
    def __init__(self):
        self.nusc = NuScenes(version='v1.0-trainval', dataroot='/data2/common/xuanyang/nuscenes', verbose=True)
        self.map_singapore_onenorth = NuScenesMap(dataroot='/data2/common/xuanyang/nuscenes', map_name='singapore-onenorth')
        self.map_singapore_hollandvillage = NuScenesMap(dataroot='/data2/common/xuanyang/nuscenes', map_name='singapore-hollandvillage')
        self.map_boston_seaport = NuScenesMap(dataroot='/data2/common/xuanyang/nuscenes', map_name='boston-seaport')
        self.map_singapore_queenstown = NuScenesMap(dataroot='/data2/common/xuanyang/nuscenes', map_name='singapore-queenstown')
        
    def get_map_instance_from_frame(self, scene_token):
        scene_info = self.nusc.get('scene', scene_token)
        log_info = self.nusc.get('log', scene_info['log_token'])
        map_name = log_info['location']
        if map_name == 'singapore-onenorth':
            map_instance = self.map_singapore_onenorth
        elif map_name == 'singapore-hollandvillage':
            map_instance = self.map_singapore_hollandvillage
        elif map_name == 'boston-seaport':
            map_instance = self.map_boston_seaport
        elif map_name == 'singapore-queenstown':
            map_instance = self.map_singapore_queenstown
        else:
            raise ValueError('Unsupported map name')
        return map_instance


    def get_ego_pose(self, frame_token):
        sample_info = self.nusc.get('sample', frame_token)
        cam_front_data = self.nusc.get('sample_data', sample_info['data']['CAM_FRONT'])
        ego_pose_info = self.nusc.get('ego_pose', cam_front_data['ego_pose_token'])
        return ego_pose_info['translation'], ego_pose_info['rotation']


    def search_lane(self, map_instance, lane_token):
        lanes = map_instance.lane
        for lane_info in lanes:
            if lane_info['token'] == lane_token:
                return lane_info
        # print (f"Error: {lane_token} not found")
        return None


    def get_nearby_lane_types(self, map_instance, scene_token, frame_token):
        ego_translation, ego_rotation = self.get_ego_pose(frame_token)
        ego_x, ego_y, ego_z = ego_translation
        road_on_point = map_instance.layers_on_point(ego_x, ego_y)
        closest_lane = map_instance.get_closest_lane(ego_x, ego_y, radius=3)
        lane_info = self.search_lane(map_instance, closest_lane)
        return ego_x, ego_y, road_on_point, lane_info


    def get_node_info(self, map_instance, node_token):
        all_node = map_instance.node
        for node in all_node:
            if node['token'] == node_token:
                return node


    def distance_cal(self,x1,y1,x2,y2):
        return ((x1-x2)**2 + (y1-y2)**2)**0.5


    def get_divider_type(self, ego_x, ego_y, map_instance, divider_segment_info):
        min_distance = 100000000
        min_node = None
        for node in divider_segment_info:
            node_info = self.get_node_info(map_instance, node['node_token'])
            distance = self.distance_cal(ego_x, ego_y, node_info['x'], node_info['y'])
            if distance < min_distance:
                min_node = node
                min_distance = distance
        return min_node


    def condition_predicate_extractor(self, conv_path, question_path, detect_info_save_path):
        yolo = YOLO('best.pt')
        with open(conv_path, 'r') as f:
            conv = json.load(f)
        with open(question_path, 'r') as f:
            questions = json.load(f)
        all_detect_info = []
        for conversation in tqdm(conv):
            id = conversation['id']
            scene_id = id.split('_')[0]
            frame_id = id.split('_')[1]
            # yolo_detection
            images = conversation['image'][:3] # cam_front cam_front_right cam_front_left
            yolo_results = set()
            yolo_result_list = []
            detected_classes = []
            for img_path in images:
                detections = yolo(img_path, verbose=False)
                for detection in detections:
                    for box in detection.boxes:
                        class_name = yolo.names[int(box.cls)]
                        detected_classes.append(class_name)
            if detected_classes:
                yolo_results.update(detected_classes)
            
            # condition_predicate_extractor
            map_instance = self.get_map_instance_from_frame(scene_id)
            ego_x, ego_y, road_on_point, lane_info = self.get_nearby_lane_types(map_instance, scene_id, frame_id)
            if road_on_point['ped_crossing'] != '':
                yolo_results.add('pedestrianCrossing')
            if road_on_point['stop_line'] != '':
                yolo_results.add('stopLine')
            if lane_info:
                if lane_info['left_lane_divider_segments']:
                    left_min_node = self.get_divider_type(ego_x, ego_y, map_instance, lane_info['left_lane_divider_segments'])
                    yolo_results.add(left_min_node['segment_type']+'_LEFT')
                if lane_info['right_lane_divider_segments']:
                    right_min_node = self.get_divider_type(ego_x, ego_y, map_instance, lane_info['right_lane_divider_segments'])
                    yolo_results.add(right_min_node['segment_type']+ '_RIGHT')
            
            # action_predicate
            question_part = questions[scene_id]["key_frames"][frame_id]["QA"]["behavior"][0]["Q"]
            answer = questions[scene_id]["key_frames"][frame_id]["QA"]["behavior"][0]["A"]
            option = get_option(question_part, answer)
            action_list = action_map(option)
            
            # save
            yolo_result_list = list(yolo_results)
            single_detect_info = {
                'image_id': id,
                'classes': yolo_result_list,
                'action': action_list,
            }
            all_detect_info.append(single_detect_info)
            with open(detect_info_save_path, 'w') as f:
                json.dump(all_detect_info, f)  
                

            
        