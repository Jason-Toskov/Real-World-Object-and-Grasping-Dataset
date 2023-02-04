import json

import numpy as np
import open3d as o3d

from src.scripts.data_loading_tools import make_grasp_data_generator
from src.scripts.utils import custom_o3d_vis

def process_grasp_ds(cfg):
    """
    Process grasp dataset and display process of incremental processing.
    Computes normals and removes plane + outliers.
    
    Parameters
    ----------
    cfg : dict
        Config dictionary.
    
    Returns
    -------
    None
    """
    data_gen = make_grasp_data_generator(cfg, use_processed=False)
    for datapoint, json_dict, rgb_img, depth_img, pcl_obj in data_gen:
        ## Point cloud processing

        # Initial cloud
        custom_o3d_vis([pcl_obj],"bbox_cam_view.json")

        # Initial crop to workspace
        min_bound = np.array(cfg['workspace']['min'])
        max_bound = np.array(cfg['workspace']['max'])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox.color = [1.0, 0, 0]
        inliers_indices_bbox = bbox.get_point_indices_within_bounding_box(pcl_obj.points)
        bbox_crop_outliers = pcl_obj.select_by_index(inliers_indices_bbox, invert=True)
        bbox_crop_outliers.paint_uniform_color([1.0, 0, 0])
        cropped_pcl = pcl_obj.crop(bbox)

        #Initial point cloud + crop box
        custom_o3d_vis([cropped_pcl, bbox_crop_outliers],"bbox_cam_view.json")

        # Plane fitting with ransac
        plane_model, inliers = cropped_pcl.segment_plane(distance_threshold=0.0075,ransac_n=100,num_iterations=1000)
        noisy_obj_cloud = cropped_pcl.select_by_index(inliers, invert=True)

        # Plane + remaining cloud
        plane_cloud = cropped_pcl.select_by_index(inliers)
        plane_cloud.paint_uniform_color([1.0, 0, 0])
        custom_o3d_vis([plane_cloud, noisy_obj_cloud],"bbox_cam_view.json")

        ## Remove outliers
        filtered_cloud, ind_filt = noisy_obj_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        outlier_cloud = noisy_obj_cloud.select_by_index(ind_filt, invert=True)
        outlier_cloud.paint_uniform_color([1.0, 0, 0])

        custom_o3d_vis([filtered_cloud, outlier_cloud],"bbox_cam_view.json")

        if True:
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
        
        custom_o3d_vis([cropped_pcl],"bbox_cam_view.json")

        # move filtered cloud to origin
        filtered_cloud.translate(-filtered_cloud.get_center())

        # final cloud
        custom_o3d_vis([filtered_cloud],"Center_object_view.json")

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)

    process_grasp_ds(cfg)