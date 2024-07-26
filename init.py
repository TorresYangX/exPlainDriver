from utils import data_prepare, train_pipeline, test_pipeline, inference_pipeline


def main():
    segment_num = -1
    pattern = 'test'
    annotation_path = "Data/video_process/conversation_bddx_{}.json".format(pattern)
    Video_folder = "Data/BDD-X/Videos/videos/"
    map_save_path = 'process_data/{}/map_ann_{}.json'.format(pattern)
    YOLO_detect_path = 'process_data/{}/{}_detected_classes.json'.format(pattern)
    vector_data_path = 'process_data/{}/{}_vectors.pkl'.format(pattern)
    weight_save_path = 'optimal_weights.npy'
    
    data_prepare(annotation_path, Video_folder, map_save_path, YOLO_detect_path, vector_data_path, segment_num)
    # train_pipeline('train_vectors.pkl', 'test_vectors.pkl', weight_save_path)
    # acc = test_pipeline(vector_data_path, weight_save_path)
    
    # test_data = np.array([0,1,0,0,0,0,0,0,0,0,0,0,1])
    # prob, action_index = inference_pipeline(test_data, weight_save_path)
    # print(prob, action_index)

if __name__ == "__main__":
    main()
    
    