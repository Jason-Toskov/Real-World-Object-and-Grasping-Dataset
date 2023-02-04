from src.scripts.data_loading_tools import make_grasp_data_generator
import json
from tqdm import tqdm

if __name__ == '__main__':
    # load config 
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    grasp_generator = make_grasp_data_generator(config, all_data = True, use_processed = True)

    i = 0
    unstable_by_object = {}
    for grasp in tqdm(grasp_generator):
        datapoint, json_dict, rgb_img, depth_img, pcl_obj = grasp

        # get the number of unstable successes
        if json_dict['success'] == 1 and json_dict['stable_success'] == 0:
            if unstable_by_object.get(json_dict['object_id']) is None:
                unstable_by_object[json_dict['object_id']] = 0
            unstable_by_object[json_dict['object_id']] += 1
            i += 1
    
    print(unstable_by_object)
    print(i)
