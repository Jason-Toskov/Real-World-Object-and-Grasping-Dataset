import json

import open3d as o3d
import tensorflow as tf
import numpy as np
from pyquaternion import Quaternion

from src.scripts.data_loading_tools import make_grasp_data_generator
from src.scripts.display_grasp_pose import init_gripper_vis, pose_to_trans_matrix
from src.scripts.utils import unpack_pose


def get_shape(x):
    """
      Gets the shape of the tensor x.
    """
    return x.get_shape().as_list()

def merge_pc_and_gripper_pc(
        pc,
        gripper_pc,
        instance_mode=0,
        pc_latent=None,
        gripper_pc_latent=None):
    """
    Merges the object point cloud and gripper point cloud and
    adds a binary auxilary feature that indicates whether each point
    belongs to the object or to the gripper.
    """

    pc_shape = get_shape(pc)
    gripper_shape = get_shape(gripper_pc)
    assert(len(pc_shape) == 3)
    assert(len(gripper_shape) == 3)
    assert(pc_shape[0] == gripper_shape[0])

    npoints = get_shape(pc)[1]
    batch_size = tf.shape(pc)[0]

    if instance_mode == 1:
        assert pc_shape[-1] == 3
        latent_dist = [pc_latent, gripper_pc_latent]
        latent_dist = tf.concat(latent_dist, 1)

    l0_xyz = tf.concat((pc, gripper_pc), 1)
    labels = [
        tf.ones(
            (get_shape(pc)[1], 1), dtype=tf.float32), tf.zeros(
            (get_shape(gripper_pc)[1], 1), dtype=tf.float32)]
    labels = tf.concat(labels, 0)
    labels = tf.expand_dims(labels, 0)
    labels = tf.tile(labels, [batch_size, 1, 1])

    if instance_mode == 1:
        l0_points = tf.concat([l0_xyz, latent_dist, labels], -1)
    else:
        l0_points = tf.concat([l0_xyz, labels], -1)

    return l0_xyz, l0_points

if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    data_gen = make_grasp_data_generator(config, all_data=True, use_processed=True)
    datapoint, json_dict, rgb_img, depth_img, pcl_obj = next(iter(data_gen))

    grasp_pos, grasp_ori = unpack_pose(json_dict['grasp_pose'])

    grasp_quart = Quaternion(x=grasp_ori[0],y=grasp_ori[1],z=grasp_ori[2],w=grasp_ori[3])
    T_grasp = grasp_quart.transformation_matrix
    T_grasp[0:3,-1] = grasp_pos
    pcl_obj.transform(np.linalg.inv(T_grasp))

    # Grasp pose
    trans_matrix = pose_to_trans_matrix(json_dict['grasp_pose'])

    gripper_model = init_gripper_vis(config)
    o3d.visualization.draw_geometries([gripper_model, pcl_obj])
    gripper_model.transform(trans_matrix)
    gripper_pcl = gripper_model.sample_points_poisson_disk(5000)

    # merge point clouds
    tf_pcl = tf.constant(pcl_obj.points, dtype=tf.float32)
    tf_grip = tf.constant(gripper_pcl.points, dtype=tf.float32)
    # insert empty batch dimension
    tf_pcl = tf.expand_dims(tf_pcl, 0)
    tf_grip = tf.expand_dims(tf_grip, 0)
    print(tf_pcl.dtype, tf_grip.dtype)

    l0_xyz, l0_points = merge_pc_and_gripper_pc(tf_pcl, tf_grip)

    breakpoint()
    print("Done!")