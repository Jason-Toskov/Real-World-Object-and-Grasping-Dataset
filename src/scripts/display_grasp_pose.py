import json
import requests

import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from src.scripts.data_loading_tools import make_grasp_data_generator
from src.scripts.utils import unpack_pose, custom_o3d_vis

def init_gripper_vis(cfg):
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

    return gripper_model

def colour_model_by_success(model, success, stable_success):
    if stable_success:
        # Green = stable success
        model.paint_uniform_color([50/255, 225/255, 50/255])
    elif success:
        # Orange = success
        model.paint_uniform_color([255/255, 165/255, 0/255])
    else:
        # Red = failure
        model.paint_uniform_color([255/255, 0/255, 0/255])
    
    return model

def pose_to_trans_matrix(pose_dict):
    pos, ori = unpack_pose(pose_dict)
    quart = Quaternion(x=ori[0],y=ori[1],z=ori[2],w=ori[3])
    trans_mat = quart.transformation_matrix
    trans_mat[0:3,-1] = pos

    return trans_mat


def show_pose(cfg, json_dict, pcl, use_processed_clouds):
    # get if grasp was successful and stable
    success = json_dict['success']
    stable_success = json_dict['stable_success']

    # create coord frame mesh
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    # Grasp pose
    trans_matrix = pose_to_trans_matrix(json_dict['grasp_pose'])

    gripper_model = init_gripper_vis(cfg)

    gripper_model.transform(trans_matrix)
    gripper_model = colour_model_by_success(gripper_model, success, stable_success)

    # if using processed clouds, center cloud at origin
    if use_processed_clouds:
        cloud_center = pcl.get_center()
        pcl.translate(-cloud_center)
        gripper_model.translate(-cloud_center)
    else:
        pcl.translate(-trans_matrix[0:3,-1])
        gripper_model.translate(-trans_matrix[0:3,-1])    
    
    return pcl, gripper_model, mesh_frame

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
        pcl,gripper_model,mesh_frame = show_pose(cfg,json_dict, pcl_obj, use_processed_clouds)
        custom_o3d_vis([pcl,gripper_model], cfg['vis_views']['center'])

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    main(cfg)