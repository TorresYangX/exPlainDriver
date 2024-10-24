from pgm.config import BDDX
from utils_bddx import bddx_prepare, train_pipeline
from test_bddx import BDDX_Test
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    pattern = 'train_filled'
    annotation_path = "Data/video_process/new_conversation_bddx_{}.json".format(pattern)
    Video_folder = "/data2/common/xuanyang/BDDX/videos/"
    map_save_path = 'Data/BDDX/process_data/{}/map_ann_{}.json'.format(pattern, pattern)
    detect_save_path = 'Data/BDDX/process_data/{}/{}_detected_classes.json'.format(pattern, pattern)
    vector_save_path = 'Data/BDDX/process_data/{}/vectors.pkl'.format(pattern)
    llm_prediction_path = 'Data/BDDX/video_process/ragdriver_kl_0.01-0.35_geo_unskew_filled/BDDX_Test_pred_action.json'
    llm_predicate_path = 'result/{}/LLM_result.json'.format(pattern)
    weight_save_path = 'weights/optimal_weights_bddx_filled.npy'    
    
    bddx_prepare(annotation_path, Video_folder, map_save_path, detect_save_path, vector_save_path, llm_prediction_path, llm_predicate_path)
    train_pipeline(vector_save_path, config=BDDX(), weight_save_path=weight_save_path)   
    BDDX_Test(weights=weight_save_path, detection_result=detect_save_path, LLM_predicate_path=llm_predicate_path)