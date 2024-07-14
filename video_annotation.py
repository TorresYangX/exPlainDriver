import json
import pandas as pd
from tqdm import tqdm

annotation_path = 'Data/BDD-X-Dataset/BDD-X-Annotations_v1.csv'
ori_annotation = pd.read_csv(annotation_path)
ori_annotation = ori_annotation.dropna(subset=['Input.Video'])
ori_annotation['Input.Video'] = ori_annotation['Input.Video'].apply(lambda x: x.split('/')[-1])


def query_annotation_csv(path, train_num, save_path):
    ori_data = json.load(open(path))[:train_num]
    data = []
    
    print("Processing data...")
    
    for item in tqdm(ori_data):
        
        original_video = item['video'].split('_')[-2].split('/')[-1]+'.mov'
        action = next(convo['value'] for convo in item['conversations'] if convo['from'] == 'gpt' and 'What is the action of ego car?' in item['conversations'][0]['value'])
        start_time, end_time = original_video_segment_map(original_video, action)
        
        if start_time == None or end_time == None:
            continue
        
        single_data = {
            'id': item['id'],
            'original_video': original_video,
            'start_time': start_time,
            'end_time': end_time,
            'process_video': item['video'],
            'action': action,
        }
        data.append(single_data)
    
    # save data to json
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Data saved to {save_path}, {len(data)} samples.")
    
    return data


def original_video_segment_map(video_name, action):
    row = ori_annotation[ori_annotation['Input.Video'] == video_name].iloc[0].dropna()
    action_columns = [col for col in row.index if 'action' in col]
    for col in action_columns:
        row[col] = row[col].strip().lower()
    action = action.strip().lower()
    target_col = None
    target_num = None
    for col in action_columns:
        if row[col] == action:
            target_col = col
            break
    if target_col:
        target_num = target_col.split('.')[1].split('action')[0]
        
    if target_num==None:
        return None, None
        
    start_col = 'Answer.'+target_num+'start'
    end_col = 'Answer.'+target_num+'end'
    try:
        start_time = row[start_col]
        end_time = row[end_col]
        return start_time, end_time
    except Exception as e:
        print(f"Error: {video_name}, {action}. Exception: {e}")
        return None, None
    
        
    