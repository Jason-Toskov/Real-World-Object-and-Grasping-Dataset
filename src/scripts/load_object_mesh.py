import json
import open3d as o3d
import numpy as np
import copy
import os

base_path = "/home/jason/object_grasp_dataset_paper/object_ds/"
objects = os.listdir(base_path)
# sort in alphabetical order
objects.sort()
for obj in objects:
    if "Resealable_Bags" in obj:
        obj_mesh = o3d.io.read_triangle_mesh(base_path+obj)
        o3d.visualization.draw_geometries([obj_mesh])