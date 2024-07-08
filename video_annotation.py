import pandas as pd
import re

action_map = {
    ("stop",): "Stop",
    ("forward",): "Keep",
    ("accelerate",): "Accelerate",
    ("brake",): "Decelerate",
    ("merge",): "Merge",
    ("right", "turn"): "MakeRightTurn",
    ("right", "lane"): "ChangeToRightLane",
    ("left", "turn"): "MakeLeftTurn",
    ("left", "lane"): "ChangeToLeftLane",
    ("pull", "over"): "PullOver",
    ("continue",): "Keep",
}


def replace_action(input_str, action_map):
    input_str_lower = input_str.lower()
    words = input_str_lower.split()
    for i in range(len(words)):
        for keys, value in action_map.items():
            if all(key in words[i:i+len(keys)] for key in keys):
                return value
    return input_str

def query_annotation_csv(path, video_index):
    table = pd.read_csv(path)
    table = table.dropna(subset=['Input.Video'])
    row = table[table['Input.Video'].str.contains(video_index)].iloc[0].dropna()
    result_dict = row.to_dict()
    return result_dict

    
if __name__=="__main__":
    input_str1 = "The car will turn right and then accelerate"
    result1 = replace_action(input_str1, action_map)
    print(result1) 
    