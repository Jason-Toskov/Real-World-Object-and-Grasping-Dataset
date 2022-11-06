import numpy as np

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