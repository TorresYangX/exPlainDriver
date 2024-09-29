import numpy as np
from pgm.config import BDDX
from pgm.PGM import PGM
from utils_bddx import data_prepare, train_pipeline
import json

if __name__ == "__main__":    
    # eval_filled_rag_top2_v8, eval_filled, train_filled_rag_top2_v8, train_filled
    pattern = 'train_filled'
    annotation_path = "Data/video_process/new_conversation_bddx_{}.json".format(pattern)
    Video_folder = "/data2/common/xuanyang/BDDX/videos/"
    map_save_path = 'Data/BDDX/process_data/{}/map_ann_{}.json'.format(pattern, pattern)
    YOLO_detect_path = 'Data/BDDX/process_data/{}/{}_detected_classes.json'.format(pattern, pattern)
    vector_data_path = 'Data/BDDX/process_data/{}/vectors.pkl'.format(pattern)
    llm_prediction_path = 'result/{}/LLM_result.json'.format(pattern)
    weight_save_path = 'weights/optimal_weights_bddx_filled.npy'    
    
    # data_prepare(annotation_path, Video_folder, map_save_path, YOLO_detect_path, vector_data_path, llm_prediction_path, 16390)
    train_pipeline(vector_data_path, config=BDDX(), weight_save_path=weight_save_path)   
    
# =================================================================

    # weights = np.load(weight_save_path)
    
    # weights = np.array([2.78633697, 1.86091075, 2.2632774, 2.77979843, 1.05179923, 1.26638907,
    #                     1.28895522, 1.01331998, 1.23938798, 1.01627995, 1.01072997, 1.09693833,
    #                     1.25381841, 1.19128857, 1.03773989, 1.08546756, 
    #                     10.02323034, 7.06097, 7.06392999, 7.06392999, 7.0569, 7.06392999, 7.06392999, 7.06281999, 7.06392999, 7.06392999, 
    #                     6.29800786, 6.88332146, 6.95238059, 7.09530089, 7.05246, 10.77237095, 10.80937099, 
    #                     10.96237095, 10.77237095,
    #                     10.01583004, 10.04950002, 10.02952003, 10.04986999, 10.06096999, 
    #                     10.03581013, 10.00954009, 10.06318999, 10.99511007, 10.05135001,
    #                     10.06096999, 10.04284002, 10.99992008, 10.97624016, 10.06059999, 10.03507024])
    
    # weights = np.array([
    #     10.02323034,  7.06097,     7.06392999,  7.06392999,  7.0569,      7.06392999,
    #     7.06392999,  7.06281999,  7.06392999,  7.06392999,  
        
    #     6.29800786,  6.88332146,  6.95238059,  6.09530089,  7.05246,    11.77237095, 8.80937099, 6.96237095,
    #     8.77237095, 
        
    #     10.51583004, 10.04950002, 10.42952003, 10.54986999, 10.06096999,
    #     10.03581013, 10.00954009, 10.06318999, 10.99511007, 10.05135001, 10.06096999,
    #     10.04284002, 10.99992008, 10.97624016, 12.06059999, 10.03507024
    #     ])
    
    # pgm = PGM(weights = weights, config=BDDX())
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

    # k_cs = 0
    # ac_cs = 1
    # dc_cs = 0
    # sp_cs = 0
    # re_cs = 0
    # st_cs = 0
    # l_cs = 1
    # r_cs = 0
    
    # k_llm = 0
    # ac_llm = 0
    # dc_llm = 1
    # sp_llm = 0
    # re_llm = 0
    # mlt_llm = 0
    # mrt_llm = 0
    # mut_llm = 0
    # mg_llm = 0
    # lp_llm = 0
    # rp_llm = 0
    # yd_llm = 0
    # ctll_llm = 0
    # ctrl_llm = 0
    # pk_llm = 0
    # po_llm = 0
    
    # data = np.array([srl, syl, yll, rll, mts, nlts, nrts,
    #                  pcs, ss, rys, sl_s, sgl,
    #                  k_cs, ac_cs, dc_cs, sp_cs, re_cs,
    #                  st_cs, l_cs, r_cs, k_llm, ac_llm, dc_llm, sp_llm, re_llm, mlt_llm, mrt_llm, mut_llm, mg_llm, lp_llm, rp_llm, yd_llm, ctll_llm, ctrl_llm, pk_llm, po_llm])
    # prob, index = pgm.infer_action_probability(data)
    # print(prob)
    