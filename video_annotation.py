import pandas as pd

video_indexs = ["06d501fd-a9ffc960.mov", "01b0505f-5f564e84.mov", "06d501fd-fd237e38.mov"]

def query_annotation_csv(path):
    result_dict = {}
    table = pd.read_csv(path)
    table = table.dropna(subset=['Input.Video'])
    for video in video_indexs:
        row = table[table['Input.Video'].str.contains(video)].iloc[0].dropna()
        row = row.drop('Input.Video')
        single_dict = row.to_dict()
        result_dict[video] = single_dict
    return result_dict
        
    