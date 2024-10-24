from pgm.config import DriveLM
from utils_drivelm import drivelm_prepare, train_pipeline
from test_drivelm import DriveLM_Test


if __name__ == "__main__":    
    pattern = 'test'
    conversation_path = 'Data/DriveLM/DriveLM_process/conversation_drivelm_{}.json'.format(pattern)
    question_path = 'Data/DriveLM/DriveLM_process/v1_1_val_nus_q_only.json'
    detect_save_path = 'Data/DriveLM/process_data_drivelm/{}/{}_detected_classes.json'.format(pattern, pattern)
    vector_save_path = 'Data/DriveLM/process_data_drivelm/{}/vectors.pkl'.format(pattern)
    # condition_vector_save_path = 'Data/DriveLM/process_data_drivelm/{}/condition_vectors.pkl'.format(pattern)
    llm_prediction_path = 'Data/DriveLM/DriveLM_process/pdce.json'
    llm_predicate_path = 'result/drivelm_pdce/LLM_result.json'
    weight_save_path = 'weights/optimal_weights_drivelm_rag_pdce.npy'    
    
    
    drivelm_prepare(conversation_path, question_path, detect_save_path, 
                 vector_save_path, llm_prediction_path, llm_predicate_path)
    train_pipeline(vector_save_path, config=DriveLM(), weight_save_path=weight_save_path)   
    DriveLM_Test(weight_save_path, detect_save_path, question_path, llm_predicate_path, llm_prediction_path)