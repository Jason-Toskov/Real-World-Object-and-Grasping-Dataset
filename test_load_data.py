import os
import glob
import json

import open3d as o3d
# from open3d.geometry import estimate_normals
import numpy as np
from tqdm import tqdm

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from utils import unpack_pose

FILE_DIR = './'
DATA_DIR = 'grasp_ds/'
SAVE_DIR = 'point_clouds_cropped_with_normals/'

DISPLAY_GRIPPER = True

def main():
    json_dirs = sorted(glob.glob(FILE_DIR + DATA_DIR + 'json_files/*/'))
    rgb_dirs = sorted(glob.glob(FILE_DIR + DATA_DIR + 'rgb_images/*/'))
    depth_dirs = sorted(glob.glob(FILE_DIR + DATA_DIR + 'depth_images/*/'))
    pcl_dirs = sorted(glob.glob(FILE_DIR + DATA_DIR + SAVE_DIR + '*/'))

    for json_f, rgb_f, depth_f, pcl_f in zip(json_dirs, rgb_dirs, depth_dirs, pcl_dirs):
        json_dirs_2 = sorted(os.listdir(json_f))
        rgb_dirs_2 = sorted(os.listdir(rgb_f))
        depth_dirs_2 = sorted(os.listdir(depth_f))
        pcl_dirs_2 = sorted(os.listdir(pcl_f))
        for json_dir, rgb, depth, pcl in zip(json_dirs_2, rgb_dirs_2, depth_dirs_2, pcl_dirs_2):

            with open(json_f+json_dir, 'r') as json_file:
                json_dict = json.load(json_file)

                success = json_dict['success']
                stable_success = json_dict['stable_success']

                # Object
                print('Cloud dir:',pcl_f+pcl)
                pcl_obj = o3d.io.read_point_cloud(pcl_f+pcl)

                # Gripper
                grasp_pos, grasp_ori = unpack_pose(json_dict['grasp_pose'])

                grasp_quart = Quaternion(x=grasp_ori[0],y=grasp_ori[1],z=grasp_ori[2],w=grasp_ori[3])
                T_grasp = grasp_quart.transformation_matrix
                T_grasp[0:3,-1] = grasp_pos

                if DISPLAY_GRIPPER:
                    ## Get gripper model
                    gripper_model = o3d.io.read_triangle_mesh('./2F85_Opened_20190924.PLY')
                    gripper_model.compute_vertex_normals()
                    gripper_bbox = gripper_model.get_axis_aligned_bounding_box()
                    gripper_z = gripper_bbox.max_bound[2] - gripper_bbox.min_bound[2]
                    gripper_model.scale(0.075/gripper_z, center=gripper_model.get_center())
                    gripper_model.translate(-gripper_model.get_center())
                    
                    gripper_bbox = gripper_model.get_axis_aligned_bounding_box()
                    gripper_box_trans = gripper_bbox.get_center()
                    gripper_model.translate(-gripper_box_trans)
                    gripper_bbox = gripper_model.get_axis_aligned_bounding_box()
                    gripper_model.translate(np.array([0,-gripper_bbox.min_bound[1]-0.128,0]))
                    r1 = R.from_euler('zx', (-90,90), degrees=True)
                    gripper_model.rotate(r1.as_matrix(), center = [0,0,0])
                    gripper_model.transform(T_grasp)

                    to_display = gripper_model
                else:
                    grasp_pos_arrow = o3d.geometry.TriangleMesh.create_arrow(0.005, 0.0075, 0.05, 0.04)
                    grasp_arrow_bbox = grasp_pos_arrow.get_axis_aligned_bounding_box()
                    grasp_pos_arrow.translate(np.array([0,0,-grasp_arrow_bbox.max_bound[2]]))
                    r1 = R.from_euler('y', (90), degrees=True)
                    grasp_pos_arrow.rotate(r1.as_matrix(), center = [0,0,0])
                    grasp_pos_arrow.compute_vertex_normals()
                    grasp_pos_arrow.transform(T_grasp)

                    to_display = grasp_pos_arrow
                
                if stable_success:
                    to_display.paint_uniform_color([50/255, 225/255, 50/255])
                elif success:
                    to_display.paint_uniform_color([255/255, 165/255, 0/255])
                else:
                    to_display.paint_uniform_color([255/255, 0/255, 0/255])

                # test view from gripper pose
                grasp_quart = Quaternion(x=grasp_ori[0],y=grasp_ori[1],z=grasp_ori[2],w=grasp_ori[3])
                T_grasp = grasp_quart.transformation_matrix
                T_grasp[0:3,-1] = grasp_pos
                pcl_obj.transform(np.linalg.inv(T_grasp))
                to_display.transform(np.linalg.inv(T_grasp))

                mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                o3d.visualization.draw_geometries([pcl_obj, mesh_frame, to_display])


if __name__ == "__main__":

    main()