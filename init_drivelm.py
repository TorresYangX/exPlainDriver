import numpy as np
from pgm.config import DriveLM
from pgm.PGM_drivelm import PGM
from utils import data_prepare, train_pipeline, map_LLM_pred
from utils_drivelm import pred_action_predicate_extractor
import json

if __name__ == "__main__":    
    pattern = 'train'
    map_save_path = 'process_data/{}/map_ann_{}.json'.format(pattern, pattern)
    YOLO_detect_path = 'process_data_drivelm/{}/{}_detected_classes.json'.format(pattern, pattern)
    vector_data_path = 'process_data_drivelm/{}/vectors.pkl'.format(pattern)
    pred_path = 'DriveLM_process/DrivingLM_Test_pred_pdce.json'
    question_path = 'DriveLM_process/v1_1_val_nus_q_only.json'
    llm_prediction_path = 'result/drivelm_pdce/LLM_result.json'
    weight_save_path = 'optimal_weights_drivelm_llm_rule.npy'    
    
    pred_action_predicate_extractor(pred_path, question_path, llm_prediction_path)
    
    # data_prepare(annotation_path, Video_folder, map_save_path, YOLO_detect_path, vector_data_path, llm_prediction_path, 16390)
    # train_pipeline(vector_data_path, config=DriveLM(), weight_save_path=weight_save_path)   
    
    # # ==================================================================
    # weight = np.load(weight_save_path)
    
    # weight = np.array([ 3.99999996,  4.74662582,  4.26896812,  4.68809649,  3.53423538,  3.75047566, 5.59472847, 
    #                    17.86671967, 17.8794396,  17.8794396,  17.8794396,  17.8794396,
    #                     17.8794396,  17.85823962, 17.8794396,  17.8794396,  17.8794396,  17.8794396,
                        
    #                     13.70061279, 13.52688565, 13.58688286, 13.49360417, 13.65220948, 13.56018218, 13.40880087, 
    #                     17.8794396,  17.8794396,  17.8794396,  17.8794396,  17.8794396, 17.8794396,  17.8794396 ])
                        
    # pgm = PGM(weights = weight, config=DriveLM())
    
    # normal = 0
    # fast = 0
    # slow = 0
    # stop = 1
    # left = 0
    # right = 0
    # straight = 1
    
    # srl = 0
    # rlal = 0
    # nlts = 0
    # nrts = 0
    # ss = 0
    
    # ddwll = 1
    # ddwlr = 0
    # sswll = 0
    # sswlr = 0
    # dswll = 0
    # dswlr = 0
    # szwll = 0
    # szwlr = 0
    # ssyll = 0
    # ssylr = 0
    
    # normal_cs = 0
    # fast_cs = 1
    # slow_cs = 0
    # stop_cs = 0
    # left_cs = 0
    # right_cs = 0
    # straight_cs = 1
    
    # normal_llm = 0
    # fast_llm = 0
    # slow_llm = 0
    # stop_llm = 1
    # left_llm = 0
    # right_llm = 0
    # straight_llm = 1
    
    # data = np.array([srl, rlal, nlts, nrts, ss,
    #                  ddwll, ddwlr, sswll, sswlr, dswll, dswlr, szwll, szwlr, ssyll, ssylr,
    #                  normal_cs, fast_cs, slow_cs, stop_cs, left_cs, right_cs, straight_cs,
    #                  normal_llm, fast_llm, slow_llm, stop_llm, left_llm, right_llm, straight_llm])
    
    # velo_prob, dire_prob, velo_index, dire_index = pgm.infer_action_probability(data)
    # print(velo_prob, dire_prob, velo_index, dire_index)
    