import os
import glob
import json

import open3d as o3d
import numpy as np
from tqdm import tqdm

FILE_DIR = './'
DATA_DIR = 'grasp_ds/'
SAVE_DIR = 'point_clouds_cropped_with_normals/'

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
        with tqdm(total=len(json_dirs_2)) as pbar:
            for json_dir, rgb, depth, pcl in zip(json_dirs_2, rgb_dirs_2, depth_dirs_2, pcl_dirs_2):

                with open(json_f+json_dir, 'r') as json_file:
                    json_dict = json.load(json_file)
                    object_id = json_dict['object_id']

                    ## Point cloud processing

                    # Load cloud
                    pcl_obj = o3d.io.read_point_cloud(pcl_f+pcl)

                    # Initial crop to workspace
                    min_bound = np.array([-0.7, -0.25, -1])
                    max_bound = np.array([-0.3, 0.25, 1])
                    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                    cropped_pcl = pcl_obj.crop(bbox)

                    # Plane fitting with ransac
                    plane_model, inliers = cropped_pcl.segment_plane(distance_threshold=0.0075,ransac_n=100,num_iterations=1000)
                    outlier_cloud = cropped_pcl.select_by_index(inliers, invert=True)

                    ## Remove outliers
                    filtered_cloud_1, ind_filt1 = outlier_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
                    filtered_cloud_2, ind_filt2 = filtered_cloud_1.remove_statistical_outlier(nb_neighbors=10, std_ratio=2)


                    ## Get normals of pcl
                    # move object to center of bounding box
                    obj_bbox = filtered_cloud_2.get_axis_aligned_bounding_box()
                    cropped_pcl.translate(-obj_bbox.get_center())
                    cropped_pcl.translate(np.array([0,0,0.05]))

                    #normal estimation
                    cropped_pcl.estimate_normals(fast_normal_computation=True,
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=30))
                    cropped_pcl.normalize_normals()
                    cropped_pcl.orient_normals_towards_camera_location(camera_location=np.array([0,0,0]))
                    # flip normals
                    cropped_pcl.normals = o3d.utility.Vector3dVector(-np.asarray(cropped_pcl.normals))

                    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                    # o3d.visualization.draw_geometries([cropped_pcl,mesh_frame])
                    
                    cropped_pcl.translate(np.array([0,0,-0.05]))
                    cropped_pcl.translate(obj_bbox.get_center())
                    outlier_cloud_norms = cropped_pcl.select_by_index(inliers, invert=True)
                    filtered_cloud_1_norms  = outlier_cloud_norms.select_by_index(ind_filt1)
                    filtered_cloud_2_norms  = filtered_cloud_1_norms.select_by_index(ind_filt2)

                    # # display filtered cloud 2
                    # o3d.visualization.draw_geometries([filtered_cloud_2_norms, mesh_frame])


                    # Save cropped cloud
                    save_path = FILE_DIR + DATA_DIR + SAVE_DIR + str(object_id) + '/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)

                    o3d.io.write_point_cloud(save_path+pcl, filtered_cloud_2_norms)

                    pbar.update(1)

if __name__ == "__main__":

    main()