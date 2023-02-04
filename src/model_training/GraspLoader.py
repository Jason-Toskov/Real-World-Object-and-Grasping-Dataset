import os
import random
import math
import json
import copy

import numpy as np
import torch.utils.data as data
import torch
import open3d as o3d

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from src.scripts.utils import unpack_pose

def merge_pcl_and_gripper_pcl(pcl, gripper_pcl):
    # Cloud is shape (N, C)
    xyz = torch.concat((pcl, gripper_pcl), axis=0)
    labels = [torch.ones((pcl.shape[0], 1)), torch.zeros((gripper_pcl.shape[0], 1))]
    labels = torch.concat(labels, 0)
    points = torch.concat([xyz, labels], axis=-1)

    return xyz, points

def pcl_to_array(pcl, normals=True):
    if normals:
        pcl_array = np.asarray(pcl.points)
        pcl_array = np.concatenate((pcl_array, np.asarray(pcl.normals)), axis=1)
    else:
        pcl_array = np.asarray(pcl.points)
    return pcl_array

def get_trans_matrix(grasp_pos, grasp_ori):
    grasp_quart = Quaternion(x=grasp_ori[0],y=grasp_ori[1],z=grasp_ori[2],w=grasp_ori[3])
    T_grasp = grasp_quart.transformation_matrix
    T_grasp[0:3,-1] = grasp_pos
    return T_grasp

def get_norm_vals(point_set):
    mean_point = np.expand_dims(np.mean(point_set, axis=0), 0)
    point_set -= mean_point
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)

    return mean_point, dist

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class GraspingDataset(data.Dataset):
    def __init__(self, args, cfg, split='train', obj_id=None, return_success=False, gripper_pcl=None):
        
        self.cfg = cfg
        self.npoints = args.num_points
        self.root = cfg['dirs']['file_dir']+cfg['dirs']['grasp_data_dir']
        self.seed = args.seed
        self.json_dir =  cfg['grasp_ds_data']['data_type_locs']['json']
        self.datapoints = []
        self.split = split
        self.train_frac = args.train_frac
        self.view_from_grasp = args.view_from_grasp
        self.use_normals = args.use_normals
        self.cross_val_num = args.cross_val_num
        self.cross_val_k = args.cross_val_k
        self.unstable_is_success = args.unstable_is_success
        self.uniform_sample = args.uniform_sample
        self.object = obj_id
        self.return_success = return_success
        self.gripper_pcl = gripper_pcl

        if gripper_pcl is not None and self.use_normals:
            grip_tmp = np.asarray(gripper_pcl.points)
            self.grip_points = np.zeros((grip_tmp.shape[0],6))
            grip_normals = np.asarray(gripper_pcl.normals)
            self.grip_points[:,:3] = grip_tmp
            self.grip_points[:,3:] = grip_normals
        elif gripper_pcl is not None:
            self.grip_points = np.asarray(gripper_pcl.points)
        else:
            self.grip_points = None

        matched_object_idx = []
        for path in os.listdir(self.root+self.json_dir): 
            for fname in os.listdir(self.root+self.json_dir+path):
                self.datapoints.append(path+'/'+fname)
                # This is bad but cbf fixing
                if self.object is not None:
                    if path == self.object:
                        matched_object_idx.append(1)
                    else:
                        matched_object_idx.append(0)
        print('-->',len(self.datapoints),'data points loaded')

        np.random.seed(self.seed)
        p = np.random.permutation(len(self.datapoints))
        if self.object is not None:
            self.datapoints, matched_object_idx = np.array(self.datapoints)[p], np.array(matched_object_idx)[p]
        else:
            self.datapoints = np.array(self.datapoints)[p]


        dp_mask = np.zeros(len(self.datapoints),dtype=bool)
        if self.cross_val_num == 0: # no cross val
            print("Randomly splitting data")
            split_idx = math.floor(len(self.datapoints)*self.train_frac)
            if self.split == 'train':
                dp_mask[:split_idx] = True
            elif self.split == 'test': 
                dp_mask[split_idx:] = True
            else:
                raise ValueError("split must be 'train' or 'test'")
        else: # cross val
            print("Cross val split number",self.cross_val_num)
            cross_val_size = math.floor(len(self.datapoints)/self.cross_val_k)
            eval_indices = np.array(list(range((self.cross_val_num-1)*cross_val_size,self.cross_val_num*cross_val_size)))
            if self.split == 'train':
                dp_mask[eval_indices] = True
                dp_mask = ~dp_mask
            elif self.split == 'test':
                dp_mask[eval_indices] = True
            else:
                raise ValueError("split must be 'train' or 'test'")
        
        self.datapoints = self.datapoints[dp_mask]
        if self.object is not None:
            matched_object_idx = np.array(matched_object_idx[dp_mask],dtype=bool)
            self.datapoints = self.datapoints[matched_object_idx]

        print('--> Dataset size:',len(self.datapoints))

        self.classes = [0,1]
        print("Number of classes:",len(self.classes))

    def __len__(self):
        return len(self.datapoints)
    
    def get_item_helper(self, idx):
        with open(self.root+self.json_dir+self.datapoints[idx], 'r') as json_file:
            json_dict = json.load(json_file)

        if json_dict['stable_success']:
            label = 1
        elif json_dict['success'] and not json_dict['stable_success'] and self.unstable_is_success:
            label = 1
        else:
            label = 0

        grasp_pos, grasp_ori = unpack_pose(json_dict['grasp_pose'])

        pcl_obj = o3d.io.read_point_cloud(self.root+self.cfg['grasp_ds_data']['processed_pcl_loc']+self.datapoints[idx]+'.pcd')

        pcl_trans = copy.deepcopy(pcl_obj)
        if self.view_from_grasp:
            # gripper needs no transforming as this is its default pose
            if self.gripper_pcl is not None:
                grip_cloud = copy.deepcopy(self.gripper_pcl)
            T_grasp = get_trans_matrix(grasp_pos, grasp_ori)
            pcl_trans.transform(np.linalg.inv(T_grasp))
        elif self.gripper_pcl is not None:
            T_grasp = get_trans_matrix(grasp_pos, grasp_ori)
            grip_cloud = copy.deepcopy(self.gripper_pcl)
            grip_cloud.transform(T_grasp)

        points = pcl_to_array(pcl_trans, normals=self.use_normals)
        if self.gripper_pcl is not None:
            gripper_points = pcl_to_array(grip_cloud, normals=self.use_normals)
        
        if self.uniform_sample:
            point_set = farthest_point_sample(points, self.npoints)
        else:
            choice = np.random.choice(len(points), self.npoints, replace=True)
            point_set = points[choice, :]

        if not self.view_from_grasp:
            mean_point, dist = get_norm_vals(point_set[:,0:3])
            point_set[:,0:3] = (point_set[:,0:3] - mean_point) / dist
            grasp_pos = (grasp_pos - mean_point[0,:]) / dist
            if self.gripper_pcl is not None:
                gripper_points[:,0:3] = (gripper_points[:,0:3] - mean_point) / dist

        point_set = torch.from_numpy(point_set.astype(np.float32))
        poses = torch.from_numpy(np.concatenate((grasp_pos,grasp_ori)).astype(np.float32))
        cls = torch.from_numpy(np.array([label]).astype(np.int64))
        if self.gripper_pcl is not None:
            gripper_points = torch.from_numpy(gripper_points.astype(np.float32))
        else: 
            gripper_points = None

        return pcl_obj, point_set, poses, cls.squeeze(), json_dict, self.datapoints[idx], gripper_points

        
    def return_single_cloud(self, idx):
        pcl_obj, point_set, poses, label, json_dict, data_path, gripper_points = self.get_item_helper(idx)
        return pcl_obj, json_dict, data_path

    def __getitem__(self, idx):
        pcl_obj, point_set, poses, label, json_dict, data_path, gripper_points = self.get_item_helper(idx)
        if self.gripper_pcl is not None:
            # Here is when we concat gripper + object
            _, point_set = merge_pcl_and_gripper_pcl(point_set, gripper_points)

        return point_set, poses, label, json_dict

    def collate_fn(self, data):
        pcl, poses, cls, json_dict = zip(*data)

        pcl = torch.stack(pcl, 0)
        poses = torch.stack(poses, 0)
        cls = torch.stack(cls, 0)

        return pcl, poses, cls, list(json_dict)

