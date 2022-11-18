import os
import random
import math
import json

import numpy as np
import torch.utils.data as data
import torch
import open3d as o3d

from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

def unpack_pose(pose_dict):
    pos = [
        pose_dict['pose']['position']['x'],
        pose_dict['pose']['position']['y'],
        pose_dict['pose']['position']['z'],
    ]

    ori = [
        pose_dict['pose']['orientation']['x'],
        pose_dict['pose']['orientation']['y'],
        pose_dict['pose']['orientation']['z'],
        pose_dict['pose']['orientation']['w']
    ]

    return np.array(pos), np.array(ori)

def get_norm_vals(point_set):
    mean_point = np.expand_dims(np.mean(point_set, axis=0), 0)
    point_set -= mean_point
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)

    return mean_point, dist


class GraspingDataset(data.Dataset):
    def __init__(self, args, cfg, split='train', obj_id=None):
        
        self.cfg = cfg
        self.npoints = args.num_points
        self.root = cfg['dirs']['file_dir']+cfg['dirs']['grasp_data_dir']
        self.seed = args.seed
        self.json_dir =  cfg['grasp_ds_data']['data_type_locs']['json']
        self.data_augmentation = args.data_augmentation
        self.datapoints = []
        self.split = split
        self.train_frac = args.train_frac
        self.view_from_grasp = args.view_from_grasp
        self.use_normals = args.use_normals
        self.cross_val_num = args.cross_val_num
        self.cross_val_k = args.cross_val_k
        self.unstable_is_success = args.unstable_is_success

        self.object = obj_id

        
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

        if self.object is not None:
            np.random.seed(self.seed)
            p = np.random.permutation(len(self.datapoints))
            self.datapoints, matched_object_idx = np.array(self.datapoints)[p], np.array(matched_object_idx)[p]
        else:
            np.random.seed(self.seed)
            p = np.random.permutation(len(self.datapoints))
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

    def __getitem__(self, idx):
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

        if self.view_from_grasp:
            grasp_quart = Quaternion(x=grasp_ori[0],y=grasp_ori[1],z=grasp_ori[2],w=grasp_ori[3])
            T_grasp = grasp_quart.transformation_matrix
            T_grasp[0:3,-1] = grasp_pos
            pcl_obj.transform(np.linalg.inv(T_grasp))


        if self.use_normals:
            point_tmp = np.asarray(pcl_obj.points)
            points = np.zeros((point_tmp.shape[0],6))
            normals = np.asarray(pcl_obj.normals)
            points[:,:3] = point_tmp
            points[:,3:] = normals
        else:
            points = np.asarray(pcl_obj.points)
        
        choice = np.random.choice(len(points), self.npoints, replace=True)
        point_set = points[choice, :]

        if not self.view_from_grasp:
            mean_point, dist = get_norm_vals(point_set)

            point_set = (point_set - mean_point) / dist
            grasp_pos = (grasp_pos - mean_point[0,:]) / dist

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        poses = torch.from_numpy(np.concatenate((grasp_pos,grasp_ori)).astype(np.float32))
        cls = torch.from_numpy(np.array([label]).astype(np.int64))
        
        return point_set, poses, cls.squeeze(), torch.LongTensor([json_dict['object_id']])