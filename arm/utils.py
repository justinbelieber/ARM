import numpy as np
import pyrender
import torch
import trimesh
import copy
from trimesh.voxel import VoxelGrid  as TVG
import matplotlib.pyplot as plt
from pyrender.trackball import Trackball
from rlbench.backend.const import DEPTH_SCALE
from scipy.spatial.transform import Rotation
from math import sin,cos,pi,acos,atan2
from scipy.spatial.transform import Rotation as R


SCALE_FACTOR = DEPTH_SCALE
DEFAULT_SCENE_SCALE = 2.0


def loss_weights(replay_sample, beta=1.0):
    loss_weights = 1.0
    if 'sampling_probabilities' in replay_sample:
        probs = replay_sample['sampling_probabilities']
        loss_weights = 1.0 / torch.sqrt(probs + 1e-10)
        loss_weights = (loss_weights / torch.max(loss_weights)) ** beta
    return loss_weights


def soft_updates(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def stack_on_channel(x):
    # expect (B, T, C, ...)
    return torch.cat(torch.split(x, 1, dim=1), dim=2).squeeze(1)


def normalize_quaternion(quat):
    return np.array(quat) / np.linalg.norm(quat, axis=-1, keepdims=True)


def quaternion_to_discrete_euler(quaternion, resolution):
    euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True) + 180
    assert np.min(euler) >= 0 and np.max(euler) <= 360
    disc = np.around((euler / resolution)).astype(int)
    disc[disc == int(360 / resolution)] = 0
    return disc


def discrete_euler_to_quaternion(discrete_euler, resolution):
    euluer = (discrete_euler * resolution) - 180
    return Rotation.from_euler('xyz', euluer, degrees=True).as_quat()


def point_to_voxel_index(
        point: np.ndarray, 
        voxel_size: np.ndarray, 
        coord_bounds: np.ndarray):  
    bb_mins = np.array(coord_bounds[0:3])
    bb_maxs = np.array(coord_bounds[3:])
    dims_m_one = np.array([voxel_size] * 3) - 1
    bb_ranges = bb_maxs - bb_mins
    res = bb_ranges / (np.array([voxel_size] * 3) + 1e-12) 
    voxel_indicy = np.minimum(
        np.floor((point - bb_mins) / (res + 1e-12)).astype(
            np.int32), dims_m_one)
    return voxel_indicy


def point_to_pixel_index(
        point: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray):
    point = np.array([point[0], point[1], point[2], 1])
    world_to_cam = np.linalg.inv(extrinsics)
    point_in_cam_frame = world_to_cam.dot(point)
    px, py, pz = point_in_cam_frame[:3]
    px = 2 * intrinsics[0, 2] - int(-intrinsics[0, 0] * (px / pz) + intrinsics[0, 2])
    py = 2 * intrinsics[1, 2] - int(-intrinsics[1, 1] * (py / pz) + intrinsics[1, 2])
    return px, py


def _compute_initial_camera_pose(scene):
    # Adapted from:
    # https://github.com/mmatl/pyrender/blob/master/pyrender/viewer.py#L1032
    centroid = scene.centroid
    scale = scene.scale
    if scale == 0.0:
        scale = DEFAULT_SCENE_SCALE
    s2 = 1.0 / np.sqrt(2.0)
    cp = np.eye(4)
    cp[:3, :3] = np.array([[0.0, -s2, s2], [1.0, 0.0, 0.0], [0.0, s2, s2]])
    hfov = np.pi / 6.0
    dist = scale / (2.0 * np.tan(hfov))
    cp[:3, 3] = dist * np.array([1.0, 0.0, 1.0]) + centroid
    return cp


def _from_trimesh_scene(
        trimesh_scene, bg_color=None, ambient_light=None):
    # convert trimesh geometries to pyrender geometries
    geometries = {name: pyrender.Mesh.from_trimesh(geom, smooth=False)
                  for name, geom in trimesh_scene.geometry.items()}
    # create the pyrender scene object
    scene_pr = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    # add every node with geometry to the pyrender scene
    for node in trimesh_scene.graph.nodes_geometry:
        pose, geom_name = trimesh_scene.graph[node]
        scene_pr.add(geometries[geom_name], pose=pose)
    return scene_pr


def _create_bounding_box(scene, voxel_size, res):
    l = voxel_size * res
    T = np.eye(4)
    w = 0.01
    for trans in [[0, 0, l / 2], [0, l, l / 2], [l, l, l / 2], [l, 0, l / 2]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [w, w, l], T, face_colors=[0, 0, 0, 255]))
    for trans in [[l / 2, 0, 0], [l / 2, 0, l], [l / 2, l, 0], [l / 2, l, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [l, w, w], T, face_colors=[0, 0, 0, 255]))
    for trans in [[0, l / 2, 0], [0, l / 2, l], [l, l / 2, 0], [l, l / 2, l]]:
        T[:3, 3] = np.array(trans) - voxel_size / 2
        scene.add_geometry(trimesh.creation.box(
            [w, l, w], T, face_colors=[0, 0, 0, 255]))


def create_voxel_scene(
        voxel_grid: np.ndarray,
        q_attention: np.ndarray = None,
        highlight_coordinate: np.ndarray = None,
        highlight_alpha: float = 1.0,
        voxel_size: float = 0.1,
        show_bb: bool = False,
        alpha: float = 0.5,
        start_point: np.ndarray = None,
        end_point: np.ndarray = None):
    _, d, h, w = voxel_grid.shape
    v = voxel_grid.transpose((1, 2, 3, 0))
    occupancy = v[:, :, :, -1] != 0
    alpha = np.expand_dims(np.full_like(occupancy, alpha, dtype=np.float32), -1)
    rgb = np.concatenate([(v[:, :, :, 3:6] + 1)/ 2.0, alpha], axis=-1)

    if q_attention is not None:
        q = np.max(q_attention, 0)
        q = q / np.max(q)
        show_q = (q > 0.75)
        occupancy = (show_q + occupancy).astype(bool)
        q = np.expand_dims(q - 0.5, -1)  # Max q can be is 0.9
        q_rgb = np.concatenate([
            q, np.zeros_like(q), np.zeros_like(q),
            np.clip(q, 0, 1)], axis=-1)
        rgb = np.where(np.expand_dims(show_q, -1), q_rgb, rgb)

    if highlight_coordinate is not None:
        x, y, z = highlight_coordinate
        occupancy[x, y, z] = True
        rgb[x, y, z] = [0.0, 0.0, 1.0, highlight_alpha]

    transform = trimesh.transformations.scale_and_translate(
        scale=voxel_size, translate=(0.0, 0.0, 0.0))
    trimesh_voxel_grid = TVG(
        encoding=occupancy, transform=transform)
    geometry = trimesh_voxel_grid.as_boxes(colors=rgb)
    scene = trimesh.Scene()
    scene.add_geometry(geometry)
    
    if start_point is not None and end_point is not None:
        points = np.array([start_point, end_point]) * voxel_size  # Scale points to match voxel size
        line = trimesh.load_path(points)
        scene.add_geometry(line)

    if show_bb:
        assert d == h == w
        _create_bounding_box(scene, voxel_size, d)
    return scene


def visualise_voxel(voxel_grid: np.ndarray,  #[c,w,h,l]
                    q_attention: np.ndarray = None,
                    highlight_coordinate: np.ndarray = None,
                    highlight_alpha: float = 1.0,
                    rotation_amount: float = 0.0,
                    show: bool = False,
                    voxel_size: float = 0.1,
                    offscreen_renderer: pyrender.OffscreenRenderer = None,
                    show_bb: bool = True,
                    pointA: np.ndarray = None,
                    pointB: np.ndarray = None):
    scene = create_voxel_scene(
        voxel_grid, q_attention, highlight_coordinate,
        highlight_alpha, voxel_size, show_bb,start_point=pointA,end_point=pointB)
    if show:
        scene.show()
    else:
        r = offscreen_renderer or pyrender.OffscreenRenderer(
            viewport_width=640, viewport_height=480, point_size=1.0)
        s = _from_trimesh_scene(
            scene, ambient_light=[0.8, 0.8, 0.8],
            bg_color=[1.0, 1.0, 1.0])
        cam = pyrender.PerspectiveCamera(
            yfov=np.pi / 4.0, aspectRatio=r.viewport_width/r.viewport_height)
        p = _compute_initial_camera_pose(s)
        t = Trackball(p, (r.viewport_width, r.viewport_height), s.scale, s.centroid)
        t.rotate(rotation_amount, np.array([0.0, 0.0, 1.0]))
        s.add(cam, pose=t.pose)
        color, depth = r.render(s)
        return color.copy()


def viewpoint_normlize(viewpoint_spher_coord,viewpoint_spher_coord_bounds):
    viewpoint_offset = viewpoint_spher_coord - viewpoint_spher_coord_bounds[:3]
    norm_viewpoint_spher_coord = viewpoint_offset / (np.array(viewpoint_spher_coord_bounds[3:]) - np.array(viewpoint_spher_coord_bounds[:3])+1e-6)
    return np.clip(norm_viewpoint_spher_coord,np.zeros(3,),np.array([1.0,1.0,1.0]))


def local_spher_to_world_pose(local_viewpoint_spher_coord:np.ndarray,
                                          world_to_local_rotation:np.ndarray=None,
                                          world_fixation_position:np.ndarray=np.zeros(3,)):

    absolute_r,absolute_theta,absolute_phi = tuple(local_viewpoint_spher_coord)

    absolute_theta = np.radians(absolute_theta)
    absolute_phi = np.radians(absolute_phi)
    

    if absolute_phi<0:
        absolute_phi += 2*np.pi
    elif absolute_phi > 2*np.pi:
        absolute_phi -= 2*np.pi 
    
    

    viewpoint_local_pos_x = absolute_r* sin(absolute_theta)*cos(absolute_phi)
    viewpoint_local_pos_y = absolute_r* sin(absolute_theta)*sin(absolute_phi)
    viewpoint_local_pos_z = absolute_r* cos(absolute_theta)
    ltc_trans = np.array([viewpoint_local_pos_x,viewpoint_local_pos_y,viewpoint_local_pos_z])

    viewpoint_axis_z = -ltc_trans/np.linalg.norm(ltc_trans)
    local_z = np.array([0.0,0.0,1.0])
    viewpoint_axis_x = np.cross(ltc_trans,local_z)
    viewpoint_axis_x = viewpoint_axis_x/np.linalg.norm(viewpoint_axis_x)
    viewpoint_axis_y = np.cross(viewpoint_axis_z,viewpoint_axis_x)
    ltc_rot = np.c_[viewpoint_axis_x,np.c_[viewpoint_axis_y,viewpoint_axis_z]]
    

        
    wtl_pos = world_fixation_position 

    if world_to_local_rotation is None:
        wtl_rot = np.array([1,0,0,
                            0,1,0,
                            0,0,1]).reshape(3,3) 
    else:
        wtl_rot = world_to_local_rotation
        
    

    wtc_trans = wtl_rot.dot(ltc_trans) + wtl_pos
    wtc_rot = wtl_rot.dot(ltc_rot)

    wtc_r = R.from_matrix(wtc_rot)
    wtc_quat = wtc_r.as_quat() #[x,y,z,w]
    wtc_euler = wtc_r.as_euler('xyz') #[x,y,z]
    
    wtc_pose = np.concatenate([wtc_trans,wtc_quat],axis=-1)
    

    return wtc_trans,wtc_euler,wtc_quat,wtc_pose




def _cartesian_to_spherical(target_cartesion_position:np.ndarray,):
    ltp_trans = target_cartesion_position
    current_r = np.linalg.norm(ltp_trans)
    current_theta = acos(ltp_trans[2]/current_r) 
    current_phi = atan2(ltp_trans[1],ltp_trans[0])
    if current_phi>np.pi:
        current_phi -= 2*np.pi

    current_phi = np.rad2deg(current_phi)
    current_theta = np.rad2deg(current_theta)

    return np.array([current_r,current_theta,current_phi])


def world_cart_to_local_spher(world_cartesion_position:np.ndarray,
                                    world_fixation_position:np.ndarray=None,
                                    world_to_local_rotation:np.ndarray=None):
    wtp_trans = world_cartesion_position
    
    wtl_pos = np.zeros([3,])
    wtl_rot = np.array([1,0,0,
                        0,1,0,
                        0,0,1]).reshape(3,3) 
    
    if world_to_local_rotation is not None:
        wtl_rot = world_to_local_rotation
    if world_fixation_position is not None:
        wtl_pos = world_fixation_position 
        
    

    ltp_trans = wtl_rot.T.dot(wtp_trans) - wtl_rot.T.dot(wtl_pos)
    
    local_spher_coord = _cartesian_to_spherical(target_cartesion_position=ltp_trans)
    
    return local_spher_coord
    

def cartesian_to_spherical(wtc_trans,fixation):
    wtl_pos = fixation 
    wtl_rot = np.array([1,0,0,
                        0,1,0,
                        0,0,1]).reshape(3,3) 
    ltc_trans = wtl_rot.T.dot(wtc_trans) - wtl_rot.T.dot(wtl_pos)
    

    current_r = np.linalg.norm(ltc_trans)
    current_theta = acos(ltc_trans[2]/current_r) 
    current_phi = atan2(ltc_trans[1],ltc_trans[0])
    

    if current_phi>np.pi:
        current_phi -= 2*np.pi

    current_phi = np.rad2deg(current_phi)
    current_theta = np.rad2deg(current_theta)

    

    return current_r,current_theta,current_phi



def world_cart_to_disc_local_spher(world_cartesion_position:np.ndarray,
                                        world_fixation_position:np.ndarray,
                                        world_to_local_rotation:np.ndarray=None,
                                        bounds=None,spher_res=None):
    continuous_spher = world_cart_to_local_spher(world_cartesion_position=world_cartesion_position,
                                                                        world_fixation_position=world_fixation_position,
                                                                        world_to_local_rotation=world_to_local_rotation)
    
    if bounds is None or spher_res is None:
        disc_spher = None
    else:
        bounds = np.array(bounds)
        spher_res = np.array(spher_res) 
        disc_spher = (continuous_spher - bounds[:3])//spher_res
        max_indices = (bounds[3:]-bounds[:3])//spher_res - 1
        disc_spher = np.clip(disc_spher,np.zeros([3]),max_indices)
    #assert np.all(max_indices >= disc_spher)
    # [3,]
    return disc_spher,continuous_spher


def world_cart_to_disc_world_spher(world_cartesion_position:np.ndarray,bounds=None,spher_res=None):
    continuous_spher = world_cart_to_local_spher(world_cartesion_position=world_cartesion_position,
                                                                        world_fixation_position=None,
                                                                        world_to_local_rotation=None)
    
    if bounds is None or spher_res is None:
        disc_spher = None
    else:
        bounds = np.array(bounds)
        spher_res = np.array(spher_res) 
        disc_spher = (continuous_spher - bounds[:3])//spher_res
        max_indices = (bounds[3:]-bounds[:3])//spher_res - 1
        disc_spher = np.clip(disc_spher,np.zeros([3]),max_indices)
    #assert np.all(max_indices >= disc_spher)
    # [3,]
    return disc_spher,continuous_spher



def local_gripper_action_to_world_action(local_gripper_action:np.ndarray,
                                        world_viewpoint_position:np.ndarray,
                                        world_fixation_position:np.ndarray=np.zeros(3,)):
    local_gripper_position = local_gripper_action[:3]
    local_gripper_quta = local_gripper_action[3:7]
    
    world_gripper_position,world_gripper_quat = local_pose_to_world_pose(local_point_position=local_gripper_position,
                                                                          local_point_quat=local_gripper_quta,
                                                                          world_fixation_position=world_fixation_position,
                                                                          world_viewpoint_position=world_viewpoint_position)
        
    world_gripper_action = np.concatenate([np.concatenate([world_gripper_position,world_gripper_quat],axis=-1),
                                           local_gripper_action[[-1]]],axis=-1)
    
    return world_gripper_action





def world_to_local_rotation(world_viewpoint_position:np.ndarray,
                            world_fixation_position:np.ndarray=np.zeros(3,)):

    # [1,] 
    fix2vp = world_viewpoint_position - world_fixation_position
    fix2vp = fix2vp/np.linalg.norm(fix2vp)
    rot_z = np.arctan2(fix2vp[1],fix2vp[0])
    cos_phi = np.cos(rot_z) 
    sin_phi = np.sin(rot_z)
    # [3,3]
    wtl_rot = np.array([cos_phi, -sin_phi, 0, 
                        sin_phi, cos_phi,  0,
                        0,         0,      1]).reshape(3,3)
    return wtl_rot


def local_pose_to_world_pose(local_point_position:np.ndarray,
                                    local_point_quat:np.ndarray,
                                    world_viewpoint_position:np.ndarray,
                                    world_fixation_position:np.ndarray=np.zeros(3,)):
    
    wtl_rot = world_to_local_rotation(world_fixation_position=world_fixation_position,
                                      world_viewpoint_position=world_viewpoint_position)
    wtl_trans = world_fixation_position

    ltp_rot = R.from_quat(local_point_quat).as_matrix()
    # [3,] 
    ltp_trans = local_point_position
    # [3,]
    wtp_trans = np.matmul(wtl_rot,ltp_trans) + wtl_trans
    # [3,3]
    wtp_rot = np.matmul(wtl_rot,ltp_rot)
    wtp_rot_quat = R.from_matrix(wtp_rot).as_quat()
    
    if wtp_rot_quat[-1] < 0:
        wtp_rot_quat = -wtp_rot_quat    
            
    #wtp = np.concatenate([wtp_trans,wtp_quat],axis=-1)
    return wtp_trans,wtp_rot_quat


    

def world_pose_to_local_pose(world_to_local_rotation:np.ndarray,
                                world_point_position:np.ndarray,
                                world_point_quat:np.ndarray,
                                world_to_local_trans:np.ndarray=np.zeros(3,)):
    wtp_trans = world_point_position
    wtl_trans = world_to_local_trans
    wtl_rot = world_to_local_rotation
    # [3,3]
    ltw_rot = wtl_rot.T
            
    wtp_rot = R.from_quat(world_point_quat).as_matrix()

    ltp_rot = np.matmul(ltw_rot,wtp_rot)
    ltp_rot_quat = R.from_matrix(ltp_rot).as_quat()
    ltp_rot_euler = R.from_matrix(ltp_rot).as_euler("xyz")
    
    if ltp_rot_quat[-1] < 0:
        ltp_rot_quat = -ltp_rot_quat    

    #wta_trans = gripper_coordinate  
    ltp_trans = np.matmul(ltw_rot,wtp_trans) - np.matmul(ltw_rot,wtl_trans)
    
    return ltp_trans,ltp_rot_quat,ltp_rot_euler



def world_pc_to_local_pc(world_viewpoint_position:np.ndarray,world_points:np.ndarray,
                                    world_fixation_position:np.ndarray=np.zeros(3,))->np.ndarray:

    wtl_rot = world_to_local_rotation(world_fixation_position=world_fixation_position,
                                      world_viewpoint_position=world_viewpoint_position)

    world_points = np.copy(world_points)
    p_shape = world_points.shape
    # [3,w,h] 
    world_points -= world_fixation_position[:,None,None]

    # [3,3]
    ltw_rot = wtl_rot.T
    # [3,w*h]
    world_points_flat = world_points.reshape(3,-1)

    local_points = np.matmul(ltw_rot,world_points_flat).reshape(p_shape)
    return local_points




def show_pc(pc_flat:np.ndarray,gripper_pose:np.ndarray=None,color=None):
    def plot_world_coordinate_system(ax, origin=[0,0,0], rot=None, length=1.0):


        if rot is not None:
            axis_x = rot[:,0]
            axis_y = rot[:,1]
            axis_z = rot[:,2]

            ax.quiver(origin[0],origin[1],origin[2], axis_x[0], axis_x[1], axis_x[2], color='r', label='X-axis')


            ax.quiver(origin[0],origin[1],origin[2], axis_y[0], axis_y[1], axis_y[2], color='g', label='Y-axis')


            ax.quiver(origin[0],origin[1],origin[2], axis_z[0], axis_z[1], axis_z[2], color='b', label='Z-axis')    
        else:    

            ax.quiver(origin[0],origin[1],origin[2], length, 0, 0, color='r', label='X-axis')


            ax.quiver(origin[0],origin[1],origin[2], 0, length, 0, color='g', label='Y-axis')


            ax.quiver(origin[0],origin[1],origin[2], 0, 0, length, color='b', label='Z-axis')    
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pc_flat = pc_flat.swapaxes(1,0)
    if color is not None:
        color = color.swapaxes(1,0)/255
    else:
        color = 'b'
    ax.scatter(pc_flat[:,0], pc_flat[:,1], pc_flat[:,2], c=color, marker='o')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')    
    
    plot_world_coordinate_system(ax)

    if gripper_pose is not None:
        gripper_rot_matrix = None
        if len(gripper_pose)==7:
            gripper_rot = R.from_quat(gripper_pose[3:]).as_matrix()
            gripper_rot_matrix = gripper_rot
        elif len(gripper_pose)==8:
            gripper_rot = R.from_quat(gripper_pose[3:-1]).as_matrix()
            gripper_rot_matrix = gripper_rot
        elif len(gripper_pose)==3:
            gripper_rot = R.from_quat(np.array([0,0,0,1])).as_matrix()
            gripper_rot_matrix = gripper_rot

        plot_world_coordinate_system(ax,gripper_pose[:3],gripper_rot_matrix,length=0.5)

    plt.show()


def visualize_voxels(tensor, pointA, pointB, closest_point=None):
    # tensor  [w,h,l,c]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    

    voxel_centers = tensor[:,:,:,:3].reshape(3, -1).T
    

    for i in range(tensor.shape[1]):
        for j in range(tensor.shape[2]):
            for k in range(tensor.shape[3]):
                if tensor[-1, i, j, k] == 1:  
                    center = voxel_centers[i * tensor.shape[2] * tensor.shape[3] + j * tensor.shape[3] + k]
                    ax.scatter(center[0], center[1], center[2], c='r', marker='o')
                else:  
                    center = voxel_centers[i * tensor.shape[2] * tensor.shape[3] + j * tensor.shape[3] + k]
                    ax.scatter(center[0], center[1], center[2], c='b', marker='s', alpha=0.1)  
    

    ax.scatter([pointA[0]], [pointA[1]], [pointA[2]], color='black', label='Point A', s=100, edgecolors='white')
    ax.scatter([pointB[0]], [pointB[1]], [pointB[2]], color='orange', label='Point B', s=100, edgecolors='white')
    

    ax.plot([pointA[0], pointB[0]], [pointA[1], pointB[1]], [pointA[2], pointB[2]], label='Ray from A to B')
    

    if closest_point is not None:
        ax.scatter([closest_point[0]], [closest_point[1]], [closest_point[2]], color='g', s=100, label='Closest Intersection')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def calculate_voxel_edge_length(voxel_centers):
    # Calculate distances between all pairs of voxel centers using broadcasting
    voxel_centers = voxel_centers.unsqueeze(0)
    distances = torch.cdist(voxel_centers, voxel_centers).squeeze(0)
    
    # Remove zero distances (distance from a voxel to itself)
    distances[distances == 0] = float('inf')
    
    # Find the minimum non-zero distance
    voxel_edge_length = distances.min().item()
    return voxel_edge_length
    
def parallel_ray_voxel_intersection(tensor, pointA, pointB,max_distance=0.03):
    # tensor  [w,w,l,c]

    voxel_centers = tensor[:,:,:,:3].reshape(3, -1).T


    direction = pointB - pointA
    direction = direction / torch.norm(direction)  
    length = torch.norm(pointB - pointA)
    

    vectors = voxel_centers - pointA
    

    projections = torch.matmul(vectors, direction)
    

    cross_product = torch.cross(direction.repeat(vectors.shape[0], 1), vectors)
    distances = torch.norm(cross_product, dim=1) / torch.norm(direction)
    

    valid_projections = (projections >= 0) & (projections <= length) & (distances <= max_distance)
    

    occupied_indices = valid_projections.nonzero().squeeze()
    occupied_voxels = tensor[:,:,:,-1].reshape(-1)[occupied_indices]
    occupied = occupied_voxels == 1
    

    if occupied.any():
        valid_distances = projections[occupied_indices][occupied]
        closest_idx = valid_distances.argmin()
        closest_point = voxel_centers[occupied_indices][occupied][closest_idx]
        

        distance_to_B = torch.norm(pointB - closest_point)
        return distance_to_B.item(), closest_point
    else:
        return float('inf'), None
    
    
    
    
def get_neighborhood_indices(index, m, max_shape):
    n = np.array(index)
    ranges = [np.arange(max(0, i - m), min(s, i + m + 1)) for i, s in zip(n, max_shape)]
    grid = np.array(np.meshgrid(*ranges, indexing='ij')).T.reshape(-1, len(n))
    grid = np.clip(grid, a_min=0, a_max=np.array(max_shape) - 1)
    return grid

def get_neighborhood_indices_cuda(index, m, max_shape):
    n = torch.tensor(index)
    ranges = [torch.arange(max(0, i - m), min(s, i + m + 1)) for i, s in zip(n, max_shape)]
    grid = torch.cartesian_prod(*ranges)
    grid = torch.clamp(grid, min=0, max=torch.tensor(max_shape) - 1)  
    return grid


def check_occupancy(voxel_tensor, indices):
    # [w,h,l,c]
    indices = indices.T
    occupancy_status = voxel_tensor[ indices[0], indices[1], indices[2],-1] == 1
    return occupancy_status.any()



