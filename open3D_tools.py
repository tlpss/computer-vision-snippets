import open3d as o3d
import numpy as np 
from typing import Tuple


# RGB float colors
cyan = (0.0, 1.0, 1.0)  
magenta = (1.0, 0.0, 1.0)  
yellow = (1.0, 1.0, 0.0)  


def visualize_open3D(geometry_list):
    o3d.visualization.draw_geometries(
        geometry_list,
        zoom=0.2,
        front=[1.0, 0.0, 1.0],
        lookat=[0.0, -0.3, 0.0],
        up=[0.0,0, 1.0],
    )


def create_open3d_ray(direction, origin=np.array([0, 0, 0])):
    c = 0.002
    ray_mesh = o3d.geometry.TriangleMesh.create_arrow(c, 4 * c, 1.5, 150 * c)

    rotation_Z = direction
    rotation_X = np.cross(rotation_Z, np.array([0, 1, 0]))
    rotation_Y = np.cross(rotation_Z, rotation_X)
    rotation_matrix = np.column_stack([rotation_X, rotation_Y, rotation_Z])

    ray_mesh.rotate(rotation_matrix, center=(0, 0, 0))

    ray_mesh.translate(origin)

    return ray_mesh

def create_open3D_camera(camera_in_world_pose, intrinsics, scale=0.1) -> Tuple[o3d.geometry.TriangleMesh, o3d.geometry.LineSet]:
    width, height = intrinsics[:2, 2].astype(np.int32) * 2
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
    camera_lines = o3d.geometry.LineSet.create_camera_visualization(
        width, height, intrinsics, np.linalg.inv(camera_in_world_pose), scale=scale
    )

    camera_lines.paint_uniform_color(cyan)

    return [camera_frame, camera_lines]


def create_open3D_point(position, color=(1, 0, 0), size=0.01):
    point = o3d.geometry.TriangleMesh.create_sphere(radius=size)
    point.paint_uniform_color(color)
    point.translate(position)

    return point


def create_open3D_frame(pose, size=0.1):
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    frame.transform(pose)

    return frame



if __name__ == "__main__":
    point = create_open3D_point([0, 0, 0])
    frame = create_open3D_frame(np.eye(4))
    ray = create_open3d_ray([0, 0, 1])
    camera = create_open3D_camera(np.eye(4), np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]))

    visualize_open3D([point, frame, ray, camera[0], camera[1]])
    import time
    time.sleep(5)  # Keep the window open for 5 seconds