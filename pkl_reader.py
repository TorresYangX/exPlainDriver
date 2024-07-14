import pickle
import json

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

if __name__ == "__main__":
    # data = pkl_reader('train_vectors.pkl')
    train_action_counter = action_counter('train_detected_classes.json')
    test_action_counter = action_counter('test_detected_classes.json')
    print(train_action_counter)
    print(test_action_counter)
