import os
import glob
import shutil
import json
import tf
import copy

import cv2
import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from rospy_message_converter.message_converter import convert_dictionary_to_ros_message as dict2rosmsg

from utils import unpack_pose

### PRESS [ESC] IN FIGURES TO VISUALIZE NEXT GRASP 

FILE_DIR = './'
DATA_DIR = 'grasp_ds/'

def main():
    json_dirs = sorted(glob.glob(FILE_DIR + DATA_DIR + 'json_files/*/'))
    rgb_dirs = sorted(glob.glob(FILE_DIR + DATA_DIR + 'rgb_images/*/'))
    depth_dirs = sorted(glob.glob(FILE_DIR + DATA_DIR + 'depth_images/*/'))
    pcl_dirs = sorted(glob.glob(FILE_DIR + DATA_DIR + 'point_clouds/*/'))

    for json_f, rgb_f, depth_f, pcl_f in zip(json_dirs, rgb_dirs, depth_dirs, pcl_dirs):
        json_dirs_2 = sorted(os.listdir(json_f))
        rgb_dirs_2 = sorted(os.listdir(rgb_f))
        depth_dirs_2 = sorted(os.listdir(depth_f))
        pcl_dirs_2 = sorted(os.listdir(pcl_f))
        for json_dir, rgb, depth, pcl in zip(json_dirs_2, rgb_dirs_2, depth_dirs_2, pcl_dirs_2):

            with open(json_f+json_dir, 'r') as json_file:
                json_dict = json.load(json_file)

                object_id = json_dict['object_id']
                # if object_id != 20:
                #     continue
                translation = json_dict['translation']
                rotation = json_dict['rotation']
                cam_info = dict2rosmsg('sensor_msgs/CameraInfo', json_dict['camera_info'])
                grasp_pose = dict2rosmsg('geometry_msgs/PoseStamped', json_dict['grasp_pose'])
                robot_pose = dict2rosmsg('geometry_msgs/PoseStamped', json_dict['robot_pose'])
                success = json_dict['success']
                stable_success = json_dict['stable_success']

                print('\nJSON file:\n',json_dict)
                print('\nObject ID:',object_id)
                print('\nTranslation:',translation)
                print('\nRotation:',rotation)
                print('\nCamera Info:\n',cam_info)
                print('\nGrasp Pose:\n',grasp_pose)
                print('\nRobot Pose:\n',robot_pose)
                print('\nSuccessful Grasp:',success)
                print('\nStable Grasp:',stable_success)
                print('\nObject ID',object_id)

            rgb_img = cv2.imread(rgb_f+rgb)
            depth_img = cv2.imread(depth_f+depth)
            pcl_obj = o3d.io.read_point_cloud(pcl_f+pcl)

            # breakpoint()
            pcl_max = pcl_obj.get_max_bound()
            pcl_min = pcl_obj.get_min_bound()
            pcl_mean, pcl_cov = pcl_obj.compute_mean_and_covariance()
            print('pcl max:',pcl_max,'\npcl min:',pcl_min,'\npcl mean:',pcl_mean)

            mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            mesh_sphere.compute_vertex_normals()
            mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
            mesh_sphere.translate(pcl_mean)

            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

            #### Crop method idea
            # 1. crop region to eliminate robot + ground entirely
            # 2. OPTIONAL: filter remaining lone ground pixels if needed 
            # 3. Get bounding box of remaining pcl (should be just the object with a decent chunk of the bottom cut off)
            # 4. Crop original point cloud with this bbox of the object + some leeway
            # This should give a clean pcl of the object with minimal ground + no robot

            # From there may want to do further post processing based on other papers
            # NOTE: Might want to change this to use grasp pose as well in crop process

            z_shifted_mean = np.array([pcl_mean[0],pcl_mean[1],0.])
            min_bound = np.array([-0.7, -0.25, -0.1])
            max_bound = np.array([-0.3, 0.25, 0.3])
            bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
            bbox.color = [0.7, 0.1, 0.1]
            print(bbox)

            box_vis = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)

            # Display grasp location
            grasp_pos, grasp_ori = unpack_pose(json_dict['grasp_pose'])
            grasp_pos_vis = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            grasp_pos_vis.compute_vertex_normals()
            grasp_pos_vis.paint_uniform_color([0.1, 0.7, 0.1])
            grasp_pos_vis.translate(grasp_pos)

            # grasp pose as a quaternion
            grasp_quart = Quaternion(x=grasp_ori[0],y=grasp_ori[1],z=grasp_ori[2],w=grasp_ori[3])
            T_grasp = grasp_quart.transformation_matrix
            T_grasp[0:3,-1] = grasp_pos
            print(T_grasp)

            # Robot pose
            robot_pos, robot_ori = unpack_pose(json_dict['robot_pose'])
            robot_quart = Quaternion(x=robot_ori[0],y=robot_ori[1],z=robot_ori[2],w=robot_ori[3])
            R_grasp = robot_quart.transformation_matrix
            R_grasp[0:3,-1] = robot_pos

            grasp_pos_arrow = o3d.geometry.TriangleMesh.create_arrow(0.005, 0.0075, 0.05, 0.04)
            grasp_arrow_bbox = grasp_pos_arrow.get_axis_aligned_bounding_box()
            grasp_pos_arrow.translate(np.array([0,0,-grasp_arrow_bbox.max_bound[2]]))
            r1 = R.from_euler('y', (90), degrees=True)
            grasp_pos_arrow.rotate(r1.as_matrix(), center = [0,0,0])
            grasp_pos_arrow.compute_vertex_normals()
            grasp_pos_arrow.paint_uniform_color([0.7, 0.1, 0.1])
            grasp_pos_arrow.transform(T_grasp)

            ## Get gripper model
            gripper_model = o3d.io.read_triangle_mesh('./2F85_Opened_20190924.PLY')
            gripper_model.compute_vertex_normals()
            gripper_bbox = gripper_model.get_axis_aligned_bounding_box()
            gripper_z = gripper_bbox.max_bound[2] - gripper_bbox.min_bound[2]
            gripper_model.scale(0.075/gripper_z, center=gripper_model.get_center())
            gripper_model.translate(-gripper_model.get_center())
            
            gripper_bbox = gripper_model.get_axis_aligned_bounding_box()
            gripper_box_trans = gripper_bbox.get_center()
            # gripper_box_trans[1] -= gripper_bbox.min_bound[1] #- 0.128
            gripper_model.translate(-gripper_box_trans)
            gripper_bbox = gripper_model.get_axis_aligned_bounding_box()
            gripper_model.translate(np.array([0,-gripper_bbox.min_bound[1]-0.128,0]))

            r1 = R.from_euler('zx', (-90,90), degrees=True)
            gripper_model.rotate(r1.as_matrix(), center = [0,0,0])
            
            o3d.visualization.draw_geometries([pcl_obj, mesh_sphere, mesh_frame,box_vis,grasp_pos_vis,grasp_pos_arrow,gripper_model])

            gripper_model.transform(T_grasp)
            if stable_success:
                gripper_model.paint_uniform_color([50/255, 225/255, 50/255])
            elif success:
                gripper_model.paint_uniform_color([255/255, 165/255, 0/255])
            else:
                gripper_model.paint_uniform_color([255/255, 0/255, 0/255])

            o3d.visualization.draw_geometries([pcl_obj, mesh_sphere, mesh_frame,box_vis,grasp_pos_vis,grasp_pos_arrow,gripper_model])
            
            cropped_pcl = pcl_obj.crop(bbox)
            if cropped_pcl.is_empty():
                print('Empty cloud!')
            o3d.visualization.draw_geometries([cropped_pcl, mesh_sphere, mesh_frame, bbox,grasp_pos_vis,grasp_pos_arrow])

            #### Save a point cloud
            # o3d.io.write_point_cloud("./fish_cropped.pcd", cropped_pcl)

            #### Plane fitting with ransac
            min_bound_plane = np.array([-0.7, -0.25, -1])
            max_bound_plane = np.array([-0.3, 0.25, 1])
            bbox_plane = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound_plane, max_bound=max_bound_plane)
            pcd_to_fit = pcl_obj.crop(bbox_plane)
            plane_model, inliers = pcd_to_fit.segment_plane(distance_threshold=0.0075,
                                         ransac_n=100,
                                         num_iterations=1000)

            [a, b, c, d] = plane_model
            print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

            inlier_cloud = pcd_to_fit.select_by_index(inliers)
            inlier_cloud.paint_uniform_color([1.0, 0, 0])
            outlier_cloud = pcd_to_fit.select_by_index(inliers, invert=True)


            # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
            o3d.visualization.draw_geometries([outlier_cloud])

            ## Remove outliers
            cl1, ind = outlier_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
            cl2, ind = cl1.remove_statistical_outlier(nb_neighbors=10, std_ratio=2)

            # cl, ind = outlier_cloud.remove_radius_outlier(nb_points=16, radius=0.05)

            # inlier_cloud_2 = cl.select_by_index(ind)
            # outlier_cloud_2 = cl.select_by_index(ind, invert=True)
            # outlier_cloud_2.paint_uniform_color([1.0, 0, 0])
            o3d.visualization.draw_geometries([cl2])

            ### Object mesh 
            #### Object mesh loading
            obj_mesh = o3d.io.read_triangle_mesh("./object_ds/Fish_Flakes.ply")
            obj_mesh_bbox = obj_mesh.get_axis_aligned_bounding_box()
            obj_mesh_z = obj_mesh_bbox.max_bound[2] - obj_mesh_bbox.min_bound[2]
            obj_mesh.scale(1/obj_mesh_z, center=obj_mesh.get_center())
            obj_mesh.translate(-obj_mesh.get_center())

            # Make sure the scales are normalized to a height of 1, 
            # seems to work a lot better for registration
            obj_mesh_pcd = obj_mesh.sample_points_poisson_disk(50000)
            cl2.translate(-cl2.get_center())
            cl2.scale(1/.115, cl2.get_center())

            o3d.visualization.draw_geometries([cl2, obj_mesh_pcd])

            # Match to pcl:
            # colored pointcloud registration
            # This is implementation of following paper
            # J. Park, Q.-Y. Zhou, V. Koltun,
            # Colored Point Cloud Registration Revisited, ICCV 2017
            voxel_radius = [0.1, 0.05, 0.01]
            max_iter = [2000, 2000, 2000]
            current_transformation = np.identity(4)
            print("3. Colored point cloud registration")
            for scale in range(len(voxel_radius)):
                iter = max_iter[scale]
                radius = voxel_radius[scale]
                print([iter, radius, scale])

                print("3-1. Downsample with a voxel size %f" % radius)
                source_down = cl2.voxel_down_sample(radius)
                target_down = obj_mesh_pcd.voxel_down_sample(radius)

                # o3d.visualization.draw_geometries([source_down, target_down])

                print("3-2. Estimate normal.")
                source_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
                target_down.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

                print("3-3. Applying colored point cloud registration")
                result_icp = o3d.pipelines.registration.registration_colored_icp(
                    source_down, target_down, radius, current_transformation,
                    o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=0.968000),
                    o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                    relative_rmse=1e-6,
                                                                    max_iteration=iter))
                current_transformation = result_icp.transformation
                print(result_icp)
                print(result_icp.transformation)
            draw_registration_result_original_color(cl2, obj_mesh, result_icp.transformation)

            # cv2.imshow("rgb", rgb_img)
            # cv2.imshow("depth", depth_img)
        
            # cv2.waitKey(0)
            

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])

if __name__ == "__main__":

    if not os.path.exists(FILE_DIR + DATA_DIR + 'point_clouds/'):
        os.makedirs(FILE_DIR + DATA_DIR + 'point_clouds/')

    for pcl_folder in glob.glob(FILE_DIR + DATA_DIR + 'point_clouds_*/'):
        for sf in glob.glob(pcl_folder + '*/'):
            shutil.move(sf, FILE_DIR + DATA_DIR + 'point_clouds/' + sf.split('/')[-2])
        os.rmdir(pcl_folder)

    main()