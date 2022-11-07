import json

import numpy as np
import open3d as o3d

from .data_loading_tools import make_grasp_data_generator
from .utils import custom_o3d_vis

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

        # Initial crop to workspace
        min_bound = np.array([-0.7, -0.25, -.05])
        max_bound = np.array([-0.3, 0.25, .3])
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        bbox.color = [1.0, 0, 0]
        cropped_pcl = pcl_obj.crop(bbox)

        #Initial point cloud + crop box
        o3d.visualization.draw_geometries([pcl_obj, bbox])

        # Plane fitting with ransac
        plane_model, inliers = cropped_pcl.segment_plane(distance_threshold=0.0075,ransac_n=100,num_iterations=1000)
        noisy_obj_cloud = cropped_pcl.select_by_index(inliers, invert=True)

        # Plane + remaining cloud
        plane_cloud = cropped_pcl.select_by_index(inliers)
        plane_cloud.paint_uniform_color([1.0, 0, 0])
        o3d.visualization.draw_geometries([plane_cloud, noisy_obj_cloud])

        ## Remove outliers
        filtered_cloud, ind_filt = noisy_obj_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        outlier_cloud = noisy_obj_cloud.select_by_index(ind_filt, invert=True)
        outlier_cloud.paint_uniform_color([1.0, 0, 0])

        o3d.visualization.draw_geometries([filtered_cloud, outlier_cloud])

        # move filtered cloud to origin
        filtered_cloud.translate(-filtered_cloud.get_center())

        # final cloud
        custom_o3d_vis([filtered_cloud],"Center_object_view.json")

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)

    process_grasp_ds(cfg)