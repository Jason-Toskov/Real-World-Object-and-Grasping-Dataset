import open3d as o3d
import os

base_path = "~/~/Real-World-Object-and-Grasping-Dataset/object_ds/"
objects = os.listdir(base_path)
# sort in alphabetical order
objects.sort()
for obj in objects:
    obj_mesh = o3d.io.read_triangle_mesh(base_path+obj)
    o3d.visualization.draw_geometries([obj_mesh])