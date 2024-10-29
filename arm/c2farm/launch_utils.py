import logging
from typing import List
import copy

import numpy as np
from omegaconf import DictConfig
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from rlbench.backend.const import TABLE_COORD
from yarr.envs.env import Env
from yarr.replay_buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer, ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from arm import demo_loading_utils, utils
from arm.custom_rlbench_env import CustomRLBenchEnv
from arm.preprocess_agent import PreprocessAgent
from arm.c2farm.networks import Qattention3DNet
from arm.c2farm.qattention_agent import QAttentionAgent
from arm.c2farm.qattention_stack_agent import QAttentionStackAgent
from arm.ota.aux_task.aux_reward import AuxReward

REWARD_SCALE = 100.0





def create_replay(batch_size: int, timesteps: int, prioritisation: bool,
                  save_dir: str, cameras: list, env: Env,
                  voxel_sizes, replay_size=1e5):

    trans_indicies_size = 3 * len(voxel_sizes)
    rot_and_grip_indicies_size = (3 + 1)

    observation_elements = env.observation_elements
    for cname in cameras:
        observation_elements.append(
            ObservationElement('%s_pixel_coord' % cname, (2,), np.int32))
    
    # 
    observation_elements.extend([
        ReplayElement('trans_action_indicies', (trans_indicies_size,),
                      np.int32),
        ReplayElement('rot_grip_action_indicies', (rot_and_grip_indicies_size,),
                      np.int32) 
    ])

    for depth in range(len(voxel_sizes)):
        observation_elements.append(
            ReplayElement('attention_coordinate_layer_%d' % depth, (3,), np.float32)
        )

    extra_replay_elements = [
        ReplayElement('demo', (), bool),
    ]

    replay_class = UniformReplayBuffer
    if prioritisation:
        replay_class = PrioritizedReplayBuffer
    replay_buffer = replay_class(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(replay_size),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=observation_elements,
        extra_replay_elements=extra_replay_elements
    )
    return replay_buffer


def _get_action(
        obs_tp1_dict: dict,
        rlbench_scene_bounds: List[float],   # AKA: DEPTH0_BOUNDS 
        voxel_sizes: List[int],    
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool):


    quat = utils.normalize_quaternion(obs_tp1_dict["gripper_pose"][3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)

    assert len(bounds_offset) == len(voxel_sizes) -1

    attention_coordinate = obs_tp1_dict["gripper_pose"][:3]
    
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    
    for depth, vox_size in enumerate(voxel_sizes):

        if depth > 0:
            # 
            if crop_augmentation:
                shift = bounds_offset[depth - 1] * 0.75
                attention_coordinate += np.random.uniform(-shift, shift, size=(3,))


            bounds = np.concatenate([attention_coordinate - bounds_offset[depth - 1],
                                     attention_coordinate + bounds_offset[depth - 1]])

        index = utils.point_to_voxel_index(
            obs_tp1_dict["gripper_pose"][:3], vox_size, bounds)

        trans_indicies.extend(index.tolist())

        res = (bounds[3:] - bounds[:3]) / vox_size

        attention_coordinate = bounds[:3] + res * index
        
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    gripper_open = obs_tp1_dict["low_dim_state"][0]
    grip = float(gripper_open)
    rot_and_grip_indicies.extend([int(gripper_open)])
    return (trans_indicies, 
            rot_and_grip_indicies, 
            np.concatenate([obs_tp1_dict["gripper_pose"], np.array([grip])]), 
            attention_coordinates) 


def _get_demo_entropy(
        demo: Demo,
        env: CustomRLBenchEnv,
        episode_keypoints: List[int],
        cameras: List[str],
        aux_reward:AuxReward):

    prev_action = None
    name = cameras[0]
    
    emtropy_sum = 0
    occupy_sum = 0
    interaction_step = 0
    obs_tp0 = demo[0]
    last_gripper_status = obs_tp0.gripper_open
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        obs_tp0_dict = env.extract_obs(obs_tp0, t=k, )
        obs_tp1_dict = env.extract_obs(obs_tp1, t=k, )
        action = obs_tp1_dict["gripper_pose"][:3]
        gripper_status = obs_tp1.gripper_open
        information_gain,roi_entropy_tp0,occ_ratio_tp0_1 = aux_reward.update_grid(target_point=action[:3],
                                            extrinsics_tp0=obs_tp0_dict['%s_camera_extrinsics' % name],
                                            extrinsics_tp0_1=obs_tp0_dict['%s_camera_extrinsics' % name],
                                            depth_tp0=obs_tp0_dict['%s_depth' % name] ,
                                            depth_tp0_1=obs_tp0_dict['%s_depth' % name] ,
                                            pc_tp0=obs_tp0_dict['%s_point_cloud' % name] ,
                                            pc_tp0_1=obs_tp0_dict['%s_point_cloud' % name] ,
                                            roi_size=0.25)
        #print(roi_entropy_tp0)
        if  gripper_status != last_gripper_status:
            interaction_step += 1
            emtropy_sum += roi_entropy_tp0 
            occupy_sum += occ_ratio_tp0_1
            
            
            
        obs_tp0 = obs_tp1  # Set the next obs
        last_gripper_status = gripper_status
            
        
    #entropy_mean = emtropy_sum/len(episode_keypoints)
    return emtropy_sum,occupy_sum,interaction_step
        



def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        inital_obs: Observation,
        demo: Demo,
        env: CustomRLBenchEnv,
        episode_keypoints: List[int],
        cameras: List[str],
        rlbench_scene_bounds: List[float],   # AKA: DEPTH0_BOUNDS
        voxel_sizes: List[int],
        bounds_offset: List[float],
        rotation_resolution: int,
        crop_augmentation: bool,):

    prev_action = None
    
    obs = inital_obs
    for k, keypoint in enumerate(episode_keypoints):
        obs_tp1 = demo[keypoint]
        obs_tp1_dict = env.extract_obs(obs_tp1, t=k, prev_action=prev_action)

        trans_indicies, rot_grip_indicies, action, attention_coordinates = _get_action(
            obs_tp1_dict, rlbench_scene_bounds, voxel_sizes, bounds_offset,
            rotation_resolution, crop_augmentation)

        terminal = (k == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0

        obs_dict = env.extract_obs(obs, t=k, prev_action=prev_action)
        prev_action = np.copy(action)

        others = {'demo': True}
        final_obs = {
            'trans_action_indicies': trans_indicies,
            'rot_grip_action_indicies': rot_grip_indicies,
        }

        for depth in range(len(voxel_sizes)):
            final_obs['attention_coordinate_layer_%d' % depth] = \
                attention_coordinates[depth]

        for name in cameras:
            px, py = utils.point_to_pixel_index(
                obs_tp1.gripper_pose[:3],
                obs_tp1.misc['%s_camera_extrinsics' % name],
                obs_tp1.misc['%s_camera_intrinsics' % name])
            final_obs['%s_pixel_coord' % name] = [py, px]
        others.update(final_obs)
        others.update(obs_dict)
        timeout = False
        replay.add(action, reward, terminal, timeout, **others)
        obs = obs_tp1  # Set the next obs
        


    # Final step
    obs_dict_tp1 = env.extract_obs(
        obs_tp1, t=k + 1, prev_action=prev_action)
    obs_dict_tp1.pop('wrist_world_to_cam', None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(**obs_dict_tp1)



def fill_replay(replay: ReplayBuffer,
                task: str,
                env: CustomRLBenchEnv,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                cameras: List[str],
                rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
                voxel_sizes: List[int],
                bounds_offset: List[float],
                rotation_resolution: int,
                crop_augmentation: bool,
                device="cpu"):

    logging.info('Filling replay with demos...')
    aux_reward = AuxReward(scene_bound=rlbench_scene_bounds,voxel_size=50,device=device)
    demo_keypoint_num = 0
    demos_entropy = 0
    demos_occupy = 0

    for d_idx in range(num_demos):
        raw_demo = env.env.get_demos(
            task, 1, variation_number=0, random_selection=False,
            from_episode_number=d_idx)[0]
        
        demo = copy.deepcopy(raw_demo)
        

        clip_observations = []
        clip_observations = [obs for obs in raw_demo._observations if obs.stage == "waypoint"]
        
        demo._observations = clip_observations
        
        #[print(step,obs.stage) for step,obs in enumerate(clip_observations)]
        
        episode_keypoints = demo_loading_utils.keypoint_discovery(demo)

        entropy, occupy, step = _get_demo_entropy(demo,env,episode_keypoints,cameras,aux_reward)
        demos_entropy += entropy
        demos_occupy += occupy
        demo_keypoint_num += step
        
        # clip_episode_keypoints = []
        # for pair in raw_episode_keypoints:
        #     cam_keypoint,hand_keypoint = pair
        #     hand_keypoint -= cam_keypoint
        #     hand_keypoint += (clip_episode_keypoints[-1]+1) if len(clip_episode_keypoints)!=0 else 0
        #     clip_episode_keypoints.append(hand_keypoint)
        # episode_keypoints = clip_episode_keypoints
            

        for i in range(len(demo) - 1):
            if not demo_augmentation and i > 0:
                break
            if i % demo_augmentation_every_n != 0:
                continue

            obs = demo[i]
            # If our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and i >= episode_keypoints[0]:
                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break

            _add_keypoints_to_replay(
                replay, obs, demo, env, episode_keypoints, cameras,
                rlbench_scene_bounds, voxel_sizes, bounds_offset,
                rotation_resolution, crop_augmentation)
    entropy_mean = demos_entropy/demo_keypoint_num
    occupy_mean = demos_occupy/demo_keypoint_num
    print(task,cameras[0],"entropy",entropy_mean, "occupy",occupy_mean)
    logging.info('Replay filled with {} initial transitions.'.format(replay.add_count.item()))



def create_agent(cfg: DictConfig, env, depth_0bounds=None, cam_resolution=None):

    VOXEL_FEATS = 3
    LATENT_SIZE = 64
    depth_0bounds = depth_0bounds or [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]
    cam_resolution = cam_resolution or [128, 128]

    include_prev_layer = False

    num_rotation_classes = int(360. // cfg.method.rotation_resolution)

    qattention_agents = []
    for depth, vox_size in enumerate(cfg.method.voxel_sizes):

        last = depth == len(cfg.method.voxel_sizes) - 1
        unet3d = Qattention3DNet(
            in_channels=VOXEL_FEATS + 3 + 1 + 3,
            out_channels=1,
            voxel_size=vox_size,
            out_dense=((num_rotation_classes * 3) + 2) if last else 0,
            kernels=LATENT_SIZE,
            norm=None if 'None' in cfg.method.norm else cfg.method.norm,
            dense_feats=128,
            activation=cfg.method.activation,
            low_dim_size=env.low_dim_state_len,
            include_prev_layer=include_prev_layer and depth > 0)


        qattention_agent = QAttentionAgent(
            layer=depth,
            coordinate_bounds=depth_0bounds,
            unet3d=unet3d,
            camera_names=cfg.rlbench.cameras,
            voxel_size=vox_size,
            bounds_offset=cfg.method.bounds_offset[depth - 1] if depth > 0 else None,
            image_crop_size=cfg.method.image_crop_size,
            tau=cfg.method.tau,
            lr=cfg.method.lr,
            lambda_trans_qreg=cfg.method.lambda_trans_qreg,
            lambda_rot_qreg=cfg.method.lambda_rot_qreg,
            include_low_dim_state=True,
            image_resolution=cam_resolution,
            batch_size=cfg.replay.batch_size,
            voxel_feature_size=3,
            exploration_strategy=cfg.method.exploration_strategy,
            lambda_weight_l2=cfg.method.lambda_weight_l2,
            num_rotation_classes=num_rotation_classes,
            rotation_resolution=cfg.method.rotation_resolution,
            grad_clip=0.01,
            gamma=0.99
        )
        qattention_agents.append(qattention_agent)

    rotation_agent = QAttentionStackAgent(
        qattention_agents=qattention_agents,
        rotation_resolution=cfg.method.rotation_resolution,
        camera_names=cfg.rlbench.cameras,
    )

    preprocess_agent = PreprocessAgent(pose_agent=rotation_agent)
    return preprocess_agent
