import json
import os
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import open3d as o3d
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

from src.scripts.display_grasp_pose import pose_to_trans_matrix, init_gripper_vis, colour_model_by_success
from src.scripts.utils import unpack_pose
from src.model_training.trainer import fwd_pass
from src.model_training.arg_parsing import parse_args
from src.model_training.models.pointnet2_cls_ssg_grasp_pose import PointNet2Model
from src.model_training.GraspLoader import GraspingDataset


def main(args, cfg):
    result_type_map = {
        '00': 'True Negative',
        '01': 'False Negative',
        '10': 'False Positive',
        '11': 'True Positive',
    }

    out_im_dir = 'grasp_ims_unprocessed/'
    use_unprocessed_cloud = True

    vis = o3d.visualization.Visualizer()

    for k, v in result_type_map.items():
        # make a directory for each result type
        if not os.path.exists(args.output_path+out_im_dir+v):
            os.makedirs(args.output_path+out_im_dir+v)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print('Using device:', device)

    # Init dataset
    ds_train = GraspingDataset(args, cfg, split='train')
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, collate_fn=ds_train.collate_fn)

    ds_test = GraspingDataset(args, cfg, split='test')
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=True, 
                            num_workers=args.num_workers, collate_fn=ds_test.collate_fn)

    # load model
    model = PointNet2Model(2, args.use_normals, args.view_from_grasp).to(device)
    # load weights from torch checkpoint
    model.load_state_dict(torch.load(args.output_path+'best_model.pt'))
    model.eval()

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for i in tqdm(range(len(ds_train))):
        pcl_obj, pcl, pose, target, json_dict, data_path = ds_train.get_item_helper(i)

        pcl, pose, target = pcl.unsqueeze(0).to(device), pose.unsqueeze(0).to(device), target.unsqueeze(0).to(device)

        output = fwd_pass(args, cfg, model, pcl, pose, target, device)

        pred = output.argmax(dim=1, keepdim=True).flatten().cpu().tolist()
        target = target.flatten().cpu().tolist()

        result_type = [str(p)+str(t) for p,t in zip(pred, target)]

        grasp_trans = pose_to_trans_matrix(json_dict['grasp_pose'])

        gripper_model = init_gripper_vis(cfg)
        gripper_model.transform(grasp_trans)
        gripper_model = colour_model_by_success(gripper_model, json_dict['success'], json_dict['stable_success'])

        cloud_center = pcl_obj.get_center()

        pcl_type = '/'.join(data_path.split('/')[-2:])
        pcl_dir = cfg['dirs']['file_dir']+cfg['dirs']['grasp_data_dir']+cfg['grasp_ds_data']['data_type_locs']['pcl']
        unprocessed_cloud = o3d.io.read_point_cloud(pcl_dir+pcl_type+'.pcd')

        unprocessed_cloud.translate(-cloud_center)
        gripper_model.translate(-cloud_center)

        if use_unprocessed_cloud:
            display_cloud = unprocessed_cloud
        else:
            display_cloud = pcl_obj

        for obj in [display_cloud, gripper_model]:
            vis.add_geometry(obj)
        ctr = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters(cfg['vis_views']['center'])
        ctr.convert_from_pinhole_camera_parameters(parameters)

        # Updates
        vis.update_geometry(display_cloud)
        vis.poll_events()
        vis.update_renderer()

        # Capture image
        time.sleep(0.05)
        vis.capture_screen_image(args.output_path+out_im_dir+result_type_map[result_type[0]]+'/'+str(i)+'.png')

        for obj in [display_cloud, gripper_model]:
            vis.remove_geometry(obj)

    vis.destroy_window()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    args.output_path = './src/'+args.output_path + 'grasp_view/'

    metric_log = main(args, cfg)