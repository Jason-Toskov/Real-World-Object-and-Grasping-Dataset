import json
import open3d as o3d
import numpy as np
import copy

from data_loading_tools import make_grasp_data_generator

def main(cfg):
    """
    Function to run the grasp pose registration.
    NOTE: This fuction requires the data to have been processed.

    Parameters
    ----------
    cfg : dict
        Config dictionary.
    
    Returns
    -------
    None
    """

    # make data generator
    data_gen = make_grasp_data_generator(cfg, use_processed = True)

    # iterate through data
    for datapoint, json_dict, rgb_img, depth_img, pcl_obj in data_gen:
        obj_id = str(datapoint['obj_idx'])
        obj_name = cfg['object_ds_data']['obj_grasp_id_map'][obj_id]

        ### Object mesh 
        #### Object mesh loading
        obj_ds_pth = cfg['dirs']['file_dir'] + cfg['dirs']['object_data_dir']
        obj_mesh = o3d.io.read_triangle_mesh(obj_ds_pth+obj_name+cfg['object_ds_data']['obj_ext'])
        obj_mesh_bbox = obj_mesh.get_axis_aligned_bounding_box()
        obj_mesh_z = obj_mesh_bbox.max_bound[2] - obj_mesh_bbox.min_bound[2]
        obj_mesh.scale(1/obj_mesh_z, center=obj_mesh.get_center())
        obj_mesh.translate(-obj_mesh.get_center())

        # Make sure the scales are normalized to a height of 1, 
        # seems to work a lot better for registration
        obj_mesh_pcd = obj_mesh.sample_points_poisson_disk(50000)
        pcl_obj.translate(-pcl_obj.get_center())
        # TODO: fix this to that object height is correct for every object
        pcl_obj.scale(1/cfg['object_ds_data']['obj_heights'][obj_id], pcl_obj.get_center()) # 1/height of object

        o3d.visualization.draw_geometries([pcl_obj, obj_mesh_pcd])

        ## Match to pcl:
        # http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html
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
            source_down = pcl_obj.voxel_down_sample(radius)
            target_down = obj_mesh_pcd.voxel_down_sample(radius)

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
        tmp_cloud = copy.deepcopy(pcl_obj)
        tmp_cloud.transform(result_icp.transformation)
        o3d.visualization.draw_geometries([tmp_cloud, obj_mesh])


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        cfg = json.load(f)
    main(cfg)