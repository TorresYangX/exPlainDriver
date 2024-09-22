import json
import re
import os
from openai import OpenAI
from utils_drivelm import control_signal_extractor, action_map
from drivelmAndPGM import question2option, option2description
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def get_min_max_signal(cs_info):
    if len(cs_info) == 0:
        return None, None, None, None
    min_val = min([item[0] for item in cs_info])
    max_val = max([item[0] for item in cs_info])
    avg_val = sum([item[0] for item in cs_info]) / len(cs_info)
    mid_val = (min_val + max_val) / 2
    return min_val, max_val, avg_val, mid_val

def main():
    conv_path = 'DriveLM_process/conversation_drivelm_train.json'
    conv_data = json.load(open(conv_path, 'r'))
    
    velo_action_cs = {
        'Normal':{'min': [], 'max': [], 'avg': [], 'mid': []},
        'Fast':{'min': [], 'max': [], 'avg': [], 'mid': []},
        'Slow':{'min': [], 'max': [], 'avg': [], 'mid': []},
        'Stop':{'min': [], 'max': [], 'avg': [], 'mid': []}
    }
    dire_action_cs = {
        'Left':{'min': [], 'max': [], 'avg': [], 'mid': []},
        'Right':{'min': [], 'max': [], 'avg': [], 'mid': []},
        'Straight':{'min': [], 'max': [], 'avg': [], 'mid': []}
    }
    
    for conv in conv_data:
        cs_string = conv['conversations'][-2]['value']
        cs_info = control_signal_extractor(cs_string)
        question = conv['conversations'][-4]['value']
        option_list = question2option(question)
        ans = conv['conversations'][-3]['value']
        ans_desc = option2description(option_list, ans)
        actions = action_map(ans_desc)
        for action in actions:
            if action in ['Normal', 'Fast', 'Slow', 'Stop']:
                velo_action = action
            else:
                dire_action = action
        min_speed, max_speed, avg_speed, mid_speed = get_min_max_signal(cs_info['Speed'])
        min_ori, max_ori, avg_speed, mid_speed = get_min_max_signal(cs_info['Orientation'])
        if min_speed:
            velo_action_cs[velo_action]['min'].append(min_speed)
            velo_action_cs[velo_action]['max'].append(max_speed)
            velo_action_cs[velo_action]['avg'].append(avg_speed)
            velo_action_cs[velo_action]['mid'].append(mid_speed)
        if min_ori:
            dire_action_cs[dire_action]['min'].append(min_ori)
            dire_action_cs[dire_action]['max'].append(max_ori)
            dire_action_cs[dire_action]['avg'].append(avg_speed)
            dire_action_cs[dire_action]['mid'].append(mid_speed)
    
    #save the result to json file
    with open('velo_action_cs.json', 'w') as f:
        json.dump(velo_action_cs, f, indent=4)
    with open('dire_action_cs.json', 'w') as f:
        json.dump(dire_action_cs, f, indent=4)
        

def distribute_figure(data):
    bin_width = 0.5
    data = np.array(data)
    p10 = np.percentile(data, 10)
    p90 = np.percentile(data, 90)

    # 打印结果
    print(f"中间80%的数据位于区间 [{p10}, {p90}]")
    bins = int((data.max() - data.min()) / bin_width)
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.7)
    plt.savefig('hist.png')
    
import json

def acc_test(slow_threshold, normal_threshold, Fast_threshold):
    conv_path = 'DriveLM_process/conversation_drivelm_train.json'
    conv_data = json.load(open(conv_path, 'r'))
    
    stop_num = 0
    slow_num = 0
    normal_num = 0
    fast_num = 0
    cor_stop = 0
    cor_slow = 0
    cor_normal = 0
    cor_fast = 0
    
    for conv in conv_data:
        cs_string = conv['conversations'][-2]['value']
        cs_info = control_signal_extractor(cs_string)
        question = conv['conversations'][-4]['value']
        option_list = question2option(question)
        ans = conv['conversations'][-3]['value']
        ans_desc = option2description(option_list, ans)
        actions = action_map(ans_desc)
        
        velo_action = None
        for action in actions:
            if action in ['Normal', 'Fast', 'Slow', 'Stop']:
                velo_action = action
        
        min_speed, max_speed = get_min_max_signal(cs_info['Speed'])
        pred_cs = None
        
        if velo_action == 'Stop':
            stop_num += 1
        elif velo_action == 'Slow':
            slow_num += 1
        elif velo_action == 'Normal':
            normal_num += 1
        elif velo_action == 'Fast':
            fast_num += 1
        
        if min_speed is not None:
            if min_speed <= 0.9 and max_speed <= slow_threshold:
                pred_cs = 'Stop'
                if velo_action == pred_cs:
                    cor_stop += 1
            elif 0.9 < min_speed < 3.6:
                pred_cs = 'Slow'
                if velo_action == pred_cs:
                    cor_slow += 1
            elif 3.6 <= min_speed < 4.0:
                pred_cs = 'Normal'
                if velo_action == pred_cs:
                    cor_normal += 1
            elif min_speed >= 4.0:
                pred_cs = 'Fast'
                if velo_action == pred_cs:
                    cor_fast += 1
    
    total_num = stop_num + slow_num + normal_num + fast_num
    correct_total = cor_stop + cor_slow + cor_normal + cor_fast
    total_acc = correct_total / total_num if total_num > 0 else 0
    
    return total_acc


def grid_search():
    best_acc = 0
    best_params = {}
    
    # Define the grid ranges for threshold values
    slow_threshold_range = [i for i in np.arange(0.4, 1.4, 0.2)]
    normal_threshold_range = [i for i in np.arange(3.0, 4.2, 0.2)]
    Fast_threshold_range = [i for i in np.arange(3.4, 4.6, 0.2)]
    for slow_threshold in tqdm(slow_threshold_range):
        for normal_threshold in tqdm(normal_threshold_range):
            for Fast_threshold in tqdm(Fast_threshold_range):
                total_acc = acc_test(slow_threshold, normal_threshold, Fast_threshold)
                if total_acc > best_acc:
                    best_acc = total_acc
                    best_params = {
                        'slow_threshold': slow_threshold,
                        'normal_threshold': normal_threshold,
                        'Fast_threshold': Fast_threshold
                    }
    
    print(f"Best Total Accuracy: {best_acc}")
    print(f"Best Parameters: {best_params}")
  
    

    
    
            
    
if __name__ == '__main__':
    speed_cs = json.load(open('velo_action_cs.json', 'r'))
    Normal_min = speed_cs['Normal']['min']
    Normal_max = speed_cs['Normal']['max']
    Normal_avg = speed_cs['Normal']['avg']
    Normal_mid = speed_cs['Normal']['mid']
    Fast_min = speed_cs['Fast']['min']
    Fast_max = speed_cs['Fast']['max']
    Fast_avg = speed_cs['Fast']['avg']
    Fast_mid = speed_cs['Fast']['mid']
    Slow_min = speed_cs['Slow']['min']
    Slow_max = speed_cs['Slow']['max']
    Slow_avg = speed_cs['Slow']['avg']
    Slow_mid = speed_cs['Slow']['mid']
    Stop_min = speed_cs['Stop']['min']
    Stop_max = speed_cs['Stop']['max']
    Stop_avg = speed_cs['Stop']['avg']
    Stop_mid = speed_cs['Stop']['mid']
    distribute_figure(Fast_mid)
    # grid_search()
    # main()
        
        
        