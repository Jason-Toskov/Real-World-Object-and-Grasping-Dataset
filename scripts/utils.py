import numpy as np
import open3d as o3d

def unpack_pose(pose_dict):
    """
    Unpacks a pose dictionary into a position and orientation vector.

    Parameters
    ----------
    pose_dict : dict
        Dictionary containing the position and orientation of a pose.

    Returns
    -------
    pos : np.array
        Position vector of the pose.
    ori : np.array
        Orientation vector of the pose.
    """
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

def custom_o3d_vis(objects, cam_json_pth):
    """
    Custom visualization function for open3d.

    Parameters
    ----------
    objects : list
        List of open3d objects to visualize.
    cam_json_pth : str
        Path to the camera json file.

    Returns
    -------
    None.

    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for obj in objects:
        vis.add_geometry(obj)
    ctr = vis.get_view_control()
    parameters = o3d.io.read_pinhole_camera_parameters(cam_json_pth)
    ctr.convert_from_pinhole_camera_parameters(parameters)
    vis.run()
    vis.destroy_window()