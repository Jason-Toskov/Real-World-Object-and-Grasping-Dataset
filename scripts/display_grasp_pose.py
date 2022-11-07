import json
import requests

import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from data_loading_tools import make_grasp_data_generator
from utils import unpack_pose, custom_o3d_vis

def main(cfg):
    """
    Displays the point cloud and gripper pose for grasps in the dataset.
    Can display either all grasps or a chosen grasp.
    Displys from a set camera location.
    
    Parameters
    ----------
    cfg : dict
        Config dictionary
    
    Returns
    -------
    None
    """
    # check if processed clouds should be used
    while True:
        use_processed_clouds = input("Use processed clouds? (y/n): ")
        if use_processed_clouds == 'y':
            use_processed_clouds = True
            break
        elif use_processed_clouds == 'n':
            use_processed_clouds = False
            break
        else:
            print("Invalid input. Please try again.")
    data_gen = make_grasp_data_generator(cfg, use_processed = use_processed_clouds)
    for datapoint, json_dict, rgb_img, depth_img, pcl_obj in data_gen:

        # get if grasp was successful and stable
        success = json_dict['success']
        stable_success = json_dict['stable_success']

        # create coord frame mesh
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        # Grasp pose
        grasp_pos, grasp_ori = unpack_pose(json_dict['grasp_pose'])
        grasp_quart = Quaternion(x=grasp_ori[0],y=grasp_ori[1],z=grasp_ori[2],w=grasp_ori[3])
        T_grasp = grasp_quart.transformation_matrix
        T_grasp[0:3,-1] = grasp_pos

        ## Get gripper model
        # Gripper model obtained from:
        # https://assets.robotiq.com/website-assets/support_documents/document/2F85_Opened_20190924.STEP
        gripper_model = o3d.io.read_triangle_mesh(cfg['dirs']['file_dir']+cfg['gripper']['path'])
        gripper_model.compute_vertex_normals()
        gripper_bbox = gripper_model.get_axis_aligned_bounding_box()
        gripper_z = gripper_bbox.max_bound[2] - gripper_bbox.min_bound[2]
        gripper_model.scale(cfg['gripper']['diam']/gripper_z, center=gripper_model.get_center())
        gripper_model.translate(-gripper_model.get_center())
        
        gripper_bbox = gripper_model.get_axis_aligned_bounding_box()
        gripper_box_trans = gripper_bbox.get_center()
        gripper_model.translate(-gripper_box_trans)
        gripper_bbox = gripper_model.get_axis_aligned_bounding_box()
        gripper_model.translate(np.array([0,-gripper_bbox.min_bound[1]-cfg['gripper']['grasp_offset'],0]))

        r1 = R.from_euler('zx', (-90,90), degrees=True)
        gripper_model.rotate(r1.as_matrix(), center = [0,0,0])

        gripper_model.transform(T_grasp)
        if stable_success:
            # Green = stable success
            gripper_model.paint_uniform_color([50/255, 225/255, 50/255])
        elif success:
            # Orange = success
            gripper_model.paint_uniform_color([255/255, 165/255, 0/255])
        else:
            # Red = failure
            gripper_model.paint_uniform_color([255/255, 0/255, 0/255])
        
        # if using processed clouds, center cloud at origin
        if use_processed_clouds:
            cloud_center = pcl_obj.get_center()
            pcl_obj.translate(-cloud_center)
            gripper_model.translate(-cloud_center)
            custom_o3d_vis([pcl_obj,gripper_model], cfg['vis_views']['center'])
        else:
            pcl_obj.translate(-grasp_pos)
            gripper_model.translate(-grasp_pos)
            custom_o3d_vis([pcl_obj,gripper_model], cfg['vis_views']['center'])
        

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    main(cfg)