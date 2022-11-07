import json

import numpy as np
import open3d as o3d
import os
from tqdm import tqdm

from data_loading_tools import make_grasp_data_generator

def process_grasp_ds(cfg):
    """
    Process grasp dataset and save processed clouds to disk.
    Computes normals and removes plane + outliers.
    
    Parameters
    ----------
    cfg : dict
        Config dictionary.
    
    Returns
    -------
    None
    """
    data_gen = make_grasp_data_generator(cfg, all_data=True, use_processed=False)
    for datapoint, json_dict, rgb_img, depth_img, pcl_obj in tqdm(data_gen, total=cfg['grasp_ds_data']['num_data_points']):
        ## Point cloud processing

        # Initial crop to workspace
        min_bound = np.array([-0.7, -0.25, -1])
        max_bound = np.array([-0.3, 0.25, 1])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        cropped_pcl = pcl_obj.crop(bbox)

        # Plane fitting with ransac
        plane_model, inliers = cropped_pcl.segment_plane(distance_threshold=0.0075,ransac_n=100,num_iterations=1000)
        outlier_cloud = cropped_pcl.select_by_index(inliers, invert=True)

        ## Remove outliers
        filtered_cloud, ind_filt = outlier_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)


        ## Get normals of pcl
        # move object to center of bounding box
        obj_bbox = filtered_cloud.get_axis_aligned_bounding_box()
        cropped_pcl.translate(-obj_bbox.get_center())
        cropped_pcl.translate(np.array([0,0,0.05]))

        #normal estimation
        cropped_pcl.estimate_normals(fast_normal_computation=True,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
        cropped_pcl.normalize_normals()
        cropped_pcl.orient_normals_towards_camera_location(camera_location=np.array([0,0,0]))
        # flip normals
        cropped_pcl.normals = o3d.utility.Vector3dVector(-np.asarray(cropped_pcl.normals))
        
        cropped_pcl.translate(np.array([0,0,-0.05]))
        cropped_pcl.translate(obj_bbox.get_center())
        outlier_cloud_norms = cropped_pcl.select_by_index(inliers, invert=True)
        filtered_cloud_norms  = outlier_cloud_norms.select_by_index(ind_filt)

        # Save cropped cloud
        dataset_dir = cfg['dirs']['file_dir']+cfg['dirs']['grasp_data_dir']
        save_dir = cfg['grasp_ds_data']['processed_pcl_loc'] 
        datapoint_dir = str(datapoint['obj_idx']) + '/' 
        save_path = dataset_dir + save_dir + datapoint_dir
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        pcl_name = str(datapoint['grasp_idx']) + cfg['grasp_ds_data']['data_type_exts']['pcl']
        o3d.io.write_point_cloud(save_path+pcl_name, filtered_cloud_norms)


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)

    process_grasp_ds(cfg)