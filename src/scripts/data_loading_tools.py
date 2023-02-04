import re
import json
import cv2
import open3d as o3d

# DATA_FOLDERS = ['json_files/','rgb_images/','depth_images/', 'point_clouds/']
def make_grasp_data_generator(cfg, all_data = False, use_processed = None):
    """
    Generator that yields a datapoint, json_dict, rgb_img, depth_img, pcl_obj.
    Can choose to unprocessed or processed data.
    Can also choose to load all data or just an individual datapoint.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the config file.
    all_data : bool, optional
        Whether to load all data or just an individual datapoint, by default False
    use_processed : bool, optional
        Whether to use processed data or not, by default None
    
    Yields
    -------
    tuple
        Tuple containing a datapoint, json_dict, rgb_img, depth_img, pcl_obj.
    """

    directory = cfg['dirs']['file_dir'] + cfg['dirs']['grasp_data_dir']
    data_folders = cfg['grasp_ds_data']['data_type_locs']
    if use_processed is None:
        while True:
            use_processed = input('Use processed data? (y/n): ')
            if use_processed == 'y':
                use_processed = True
                break
            elif use_processed == 'n':
                use_processed = False
                break
            else:
                print('Please enter y or n')
    if use_processed:
        data_folders['pcl'] = cfg['grasp_ds_data']['processed_pcl_loc']

    # get user input, to choose whether to load one datapoint or loop over all datapoints
    while True:
        if not all_data:
            inp = input("Enter '1' to load one datapoint, or '2' to loop over all datapoints: ")
        else:
            inp = '2'
            
        if inp == '1':
            # choose object index
            obj_idx = input("Enter object index (1-20): ")
            # check with regex if obj_idx is from 1 to 20
            if re.match(r'^[1-9]$|^1[0-9]$|^20$', obj_idx):
                obj_idx = [int(obj_idx)]
                # choose grasp index
                grasp_idx = input("Enter grasp index (0-74): ")
                # check with regex if grasp_idx is from 0 to 74
                if re.match(r'^[0-9]$|^[1-6][0-9]$|^7[0-4]$', grasp_idx):
                    grasp_idx = [int(grasp_idx)]\
                    # print selected object and grasp
                    print("Selected object index is {} and grasp index is {}".format(obj_idx[0], grasp_idx[0]))
                    break
                else:
                    print("Invalid grasp index. Please enter a number from 0 to 74.")
            else:
                print("Invalid object index. Please enter a number between 1 and 20.")
                continue
        elif inp == '2' or all_data:
            print("Loading all datapoints...")
            obj_idx = list(range(1,21))
            grasp_idx = list(range(75))
            break
        else:
            print("Invalid input")
    
    for obj in obj_idx:
        for grasp in grasp_idx:
            datapoint = {'obj_idx': obj, 'grasp_idx': grasp}
            data_dir = {k: directory + folder + str(obj) + '/' + str(grasp) for k,folder in data_folders.items()}
            # load files
            with open(data_dir['json'], 'r') as json_file:
                json_dict = json.load(json_file)
            rgb_img = cv2.imread(data_dir['rgb']+cfg['grasp_ds_data']['data_type_exts']['rgb'])
            depth_img = cv2.imread(data_dir['depth'] + cfg['grasp_ds_data']['data_type_exts']['depth'])
            pcl_obj = o3d.io.read_point_cloud(data_dir['pcl'] + cfg['grasp_ds_data']['data_type_exts']['pcl'])
            yield datapoint, json_dict, rgb_img, depth_img, pcl_obj


if __name__ == '__main__':
    # load config file
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    data_gen = make_grasp_data_generator(config)

    for datapoint, json_dict, rgb_img, depth_img, pcl_obj in data_gen:
        # print data from json dict nicely
        print("Grasp data: ")
        print(json.dumps(json_dict, indent=4))
        # print datapoint object and grasp index
        print("Object index is {} and grasp index is {}".format(datapoint['obj_idx'], datapoint['grasp_idx']))
        print('\n Press "q" to quit')
        o3d.visualization.draw_geometries([pcl_obj])
        print("Press any key to continue...")
        cv2.imshow('depth_img', depth_img)
        cv2.imshow('rgb_img', rgb_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

