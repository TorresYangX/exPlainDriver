import numpy as np
from pgm.config import BDDX
from pgm.PGM import PGM
from utils import data_prepare, train_pipeline, test_pipeline, video_snapshot
import json
import pickle

from collections import Counter

if __name__ == "__main__":    
    pattern = 'test'
    annotation_path = "Data/video_process/new_conversation_bddx_{}.json".format(pattern)
    Video_folder = "/data2/common/xuanyang/BDDX/videos/"
    map_save_path = 'process_data/{}/map_ann_{}.json'.format(pattern, pattern)
    YOLO_detect_path = 'process_data/{}/{}_detected_classes.json'.format(pattern, pattern)
    vector_data_path = 'process_data/{}/{}_vectors.pkl'.format(pattern, pattern)
    weight_save_path = 'optimal_weights.npy'
    
    data_prepare(annotation_path, Video_folder, map_save_path, YOLO_detect_path, vector_data_path, 16390)
    # train_pipeline(vector_data_path, config=BDDX(), weight_save_path=weight_save_path)
    
    
    # weights = np.array([
    #     15.51559968,  15.07370472, 15.73000304, 15.45839886,  12.28816603,  12.93708945,
    #     12.76099285,  12.08004622,  12.59233605,  12.06403711,  12.06403630,  12.03201807,
    #     12.65637727,  12.51229441,  12.08004540,  12.19210990, 30.65301903, 12.03725726,
    #     15.06927711, 15.06927711, 15.06927711, 15.06927711, 15.06927711, 
    #     34.89316792,
    #     # 34.66902898, 
    #     # 15.06927711, 
    #     15.06927711, 
    #     # 30.89316803, 
    #     5.00000000, 5.00000000,
    #     5.00000000, 5.00000000, 
    #     10.00000000,
    #     10.00000000, 10.00000000
    # ]) 
    
    # pgm = PGM(weights = weights, config=BDDX())
    
    # kp = 0
    # ac = 0
    # dc = 1
    # sp = 0
    # re = 0
    # mlt = 0
    # mrt = 0
    # mut = 0
    # mg = 0
    # lp = 0
    # rp = 0
    # yd = 0
    # ctll = 0
    # ctrl = 0
    # pk = 0
    # po = 0

    # srl = 0
    # syl = 0
    # yll = 0
    # rll = 0
    # mts = 0
    # nlts = 0
    # nrts = 0
    # pcs = 0
    # ss = 0
    # rys = 0
    # sl_s = 0
    # sgl = 0

    # k_cs = 1
    # ac_cs = 0
    # dc_cs = 0
    # sp_cs = 0
    # re_cs = 0
    # st_cs = 1
    # l_cs = 0
    # r_cs = 0

    # # Create numpy array
    # data = np.array([
    #     kp, ac, dc, sp, re, mlt, mrt, mut, 
    #     mg, lp, rp, yd, ctll, ctrl, pk, po,
    #     srl, syl, yll, rll, mts, nlts, nrts,
    #     pcs, ss, rys, sl_s, sgl,
    #     k_cs, ac_cs, dc_cs, sp_cs, re_cs,
    #     st_cs, l_cs, r_cs
    # ])
    # prob = pgm.compute_instance_probability(data)
    # print(prob)
    
    # data = np.array([srl, syl, yll, rll, mts, nlts, nrts,
    #                  pcs, ss, rys, sl_s, sgl,
    #                  k_cs, ac_cs, dc_cs, sp_cs, re_cs,
    #                  st_cs, l_cs, r_cs])
    # prob, index = pgm.infer_action_probability(data)
    # print(prob)
    
    
    # video_root = "/data2/common/xuanyang/BDDX/videos"
    # video = '2aee5704-baf98720.mov'
    # video_path = os.path.join(video_root, video)
    # video_path = os.path.abspath(video_path)
    # output_folder = 'Data/video_snapshot'
    # start_time = 0
    # end_time = 16
    # video_snapshot(video_path, output_folder, start_time, end_time)
    
    
    
    