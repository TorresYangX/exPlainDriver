import pickle
import json
import cv2
import os

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




if __name__ == "__main__":
    # # data = pkl_reader('train_vectors.pkl')
    # train_action_counter = action_counter('train_detected_classes.json')
    # print(train_action_counter)
    # # compute each action's proportion
    # total = sum(train_action_counter.values())
    # for key in train_action_counter:
    #     train_action_counter[key] /= total
    
    # prob = []
    # for key in train_action_counter:
    #     prob.append(train_action_counter[key])
        
    # print(prob)
    
    video_path_root = 'Data/BDD-X/Videos/videos/'
    output_folder = 'Data/video_snapshot'
    
    video = '2af58da3-c04e01b3.mov'
    start = 0
    end = 21
    
    video_path = os.path.join(video_path_root, video)
    video_snapshot(video_path, output_folder, start, end)