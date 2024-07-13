import json

def query_annotation_csv(path, train_num):
    ori_data = json.load(open(path))[:train_num]
    data = []
    for item in ori_data:
        single_data = {
            'id': item['id'],
            'video': item['video'],
            'action': next(convo['value'] for convo in item['conversations'] if convo['from'] == 'gpt' and 'What is the action of ego car?' in item['conversations'][0]['value'])
        }
        data.append(single_data)
    
    return data
    
        
    