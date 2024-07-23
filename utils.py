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


def video_snapshot(video_path, output_folder, interval=1):
    video_name = video_path.split('/')[-1].split('.')[0]
    output_path = os.path.join(output_folder, video_name)
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * 0.5)
    frame_count = 0
    image_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_path, f'image_{image_count:04d}.jpg')
            cv2.imwrite(image_path, frame)
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
    
    video_snapshot('Data/BDD-X/Videos/videos/1aecebd4-af1a595a.mov', 'Data/video_snapshot')