import logging
from typing import List
import copy
import numpy as np
from rlbench.backend.observation import Observation
from rlbench.demo import Demo
from yarr.replay_buffer.prioritized_replay_buffer import \
    PrioritizedReplayBuffer
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer

from arm import demo_loading_utils, utils
from arm.baselines.bc_active.bc_agent import BCAgent
from arm.custom_rlbench_env import CustomRLBenchEnv
from arm.ota.custom_rlbench_env import OtaCustomRLBenchEnv
from arm.ota.const import SAVE_OBS_KEYS,SAVE_OBS_ELEMENT_KEYS

from arm.network_utils import SiameseNet, CNNAndFcsNet
from arm.baselines.bc_active.preprocess_agent import PreprocessAgent
REWARD_SCALE = 100.0

def create_replay(batch_size: int, timesteps: int, prioritisation: bool,
                  save_dir: str, env: CustomRLBenchEnv):
    observation_elements = env.observation_elements
    
    save_obs_elm = []
    for el in observation_elements:
        if el.name in SAVE_OBS_KEYS:
            for depth in range(2):
                copy_el = copy.deepcopy(el)

                if copy_el.name == 'low_dim_state':
                    copy_el.shape = (17,)
                copy_el.name = copy_el.name + '_layer_{}'.format(depth)
                save_obs_elm.append(copy_el)

    save_obs_elm.append(ReplayElement('action_layer_0', (2,), np.float32))
    save_obs_elm.append(ReplayElement('action_layer_1', (8,), np.float32))



    replay_class = UniformReplayBuffer
    if prioritisation:
        replay_class = PrioritizedReplayBuffer
    replay_buffer = replay_class(
        save_dir=save_dir,
        batch_size=batch_size,
        timesteps=timesteps,
        replay_capacity=int(1e5),
        action_shape=(8,),
        action_dtype=np.float32,
        reward_shape=(),
        reward_dtype=np.float32,
        update_horizon=1,
        observation_elements=save_obs_elm,
        extra_replay_elements=[ReplayElement('demo', (), bool)]
    )
    return replay_buffer


def _get_nbp_action(obs_tp1:dict):
    quat = utils.normalize_quaternion(obs_tp1["gripper_pose"][3:])
    if quat[-1] < 0:
        quat = -quat
    return np.concatenate([obs_tp1["gripper_pose"][:3], quat,
                           [float(obs_tp1["low_dim_state"][0])]])
    
def _get_nbv_action(obs_dict_tp0_1,viewpoint_env_bounds: List[float]):
    
    vp_world_pos_tp0_1 = obs_dict_tp0_1["active_cam_pose"][:3]

    vp_spher_pos_tp0_1 = utils.world_cart_to_local_spher(world_cartesion_position=vp_world_pos_tp0_1)

    vp_action = (vp_spher_pos_tp0_1[1:]-np.array(viewpoint_env_bounds[1:3]))/(np.array(viewpoint_env_bounds[4:]) - np.array(viewpoint_env_bounds[1:3]))

    return vp_action
    
    


def _add_keypoints_to_replay(
        replay: ReplayBuffer,
        obs_tp0: Observation,
        demo: Demo,
        env: OtaCustomRLBenchEnv,
        episode_keypoints: List[List[int]],
        viewpoint_env_bounds: List[float], 
        device="cpu"):
    # tp0 
    obs_dict_tp0 = env.extract_obs(obs_tp0,)
    for keypoints_index, keypoints in enumerate(episode_keypoints):

        raw_tp0_1_step_index = keypoints[0]

        raw_tp1_step_index = keypoints[1]
        
        tp0_1_step_index = raw_tp0_1_step_index
        tp1_step_index = raw_tp1_step_index
        
        
        final_obs = {}
        # time point 1
        obs_dict_tp0_1 = env.extract_obs(demo[tp0_1_step_index])
        obs_dict_tp1 = env.extract_obs(demo[tp1_step_index])

        nbv_action = _get_nbv_action(obs_dict_tp0_1,viewpoint_env_bounds)
        nbp_action = _get_nbp_action(obs_dict_tp1)

        terminal = (keypoints_index == len(episode_keypoints) - 1)
        reward = float(terminal) * REWARD_SCALE if terminal else 0        

        for depth,obs in enumerate([obs_dict_tp0,obs_dict_tp0_1]):  
            for k,v in obs.items():
                if k in SAVE_OBS_KEYS:
                    final_obs[k+'_layer_{}'.format(depth)] = v          

        actions = [nbv_action,nbp_action]

        for depth in range(2):
            final_obs['action'+'_layer_{}'.format(depth)] = actions[depth]
        others = {'demo': True}

        others.update(final_obs)
        timeout = False

        replay.add(nbp_action, reward, terminal, timeout, **others)
        # Set the next obs
        obs_dict_tp0 = obs_dict_tp1


    obs_dict_tp1.pop('active_world_to_cam', None)


    save_obs_tp1 = {}
    for k,v in obs_dict_tp1.items():
        if k in SAVE_OBS_KEYS:

            for depth in range(2):
                save_obs_tp1[k+'_layer_{}'.format(depth)] = v

    final_obs.update(save_obs_tp1)
    replay.add_final(**final_obs)


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[List[int]]:

    episode_keypoints = []
    transition_keypoints = []
    prev_transition_index = 0
    prev_stage = 'viewpoint'
    demo_length = len(demo)
    for step, obs in enumerate(demo):
        stage = obs.stage
        current_transition_index = obs.transition_index
        if stage is None and current_transition_index==0:
            continue
        # 
        if current_transition_index == prev_transition_index and stage is not None:

            if stage != prev_stage:
                transition_keypoints.append(step-1)
                prev_stage = stage
                continue
        elif prev_transition_index !=0 :

            transition_keypoints.append(step-1)
            assert len(transition_keypoints) == 2
            episode_keypoints.append(transition_keypoints)
            transition_keypoints = []
            prev_stage = 'viewpoint'
            prev_transition_index = current_transition_index
        else:
            prev_stage = 'viewpoint'
            prev_transition_index = current_transition_index
                
    logging.debug('Found %d keypoints.' % len(episode_keypoints),
                  episode_keypoints)
    return episode_keypoints



def fill_replay(replay: ReplayBuffer,
                task: str,
                env: OtaCustomRLBenchEnv,
                num_demos: int,
                demo_augmentation: bool,
                demo_augmentation_every_n: int,
                viewpoint_env_bounds: List[float],
                device="cpu"):

    logging.info('Filling replay with demos...')
    
    
    for d_idx in range(num_demos):

        demo = env.env.get_demos(
            task, 1, variation_number=0, random_selection=False,
            from_episode_number=d_idx)[0]

        episode_keypoints = keypoint_discovery(demo)
        
        for tp0_index in range(len(demo) - 1):
            if not demo_augmentation and tp0_index > 0 :
                break
            
            # If our starting point is past one of the keypoints, then remove it
            while len(episode_keypoints) > 0 and tp0_index >= episode_keypoints[0][0]:

                episode_keypoints = episode_keypoints[1:]
            if len(episode_keypoints) == 0:
                break
                

            if tp0_index % demo_augmentation_every_n != 0  and  tp0_index != episode_keypoints[0][0]:
                continue

            if tp0_index >= episode_keypoints[0][0] and tp0_index < episode_keypoints[0][1]:
                continue
            obs_tp0 = demo[tp0_index]
            

            augmented_episode_keypoints = copy.deepcopy(episode_keypoints)
            for tp0_1_index in range(episode_keypoints[0][0],episode_keypoints[0][1]):
                # 
                if not demo_augmentation and tp0_1_index > episode_keypoints[0][0]:
                    break

                if tp0_1_index % demo_augmentation_every_n != 0 and tp0_1_index != episode_keypoints[0][0]:
                    continue

                augmented_episode_keypoints[0][0] = tp0_1_index
                _add_keypoints_to_replay(
                    replay, obs_tp0, demo, env, augmented_episode_keypoints,viewpoint_env_bounds,device)
            
    logging.info('Replay filled with {} initial transitions.'.format(replay.add_count.item()))



def create_agent(camera_name: str,
                 activation: str,
                 lr: float,
                 weight_decay: float,
                 image_resolution: list,
                 grad_clip: float,
                 low_dim_state_len: int):

    siamese_net = SiameseNet(
        input_channels=[3, 3],
        filters=[16],
        kernel_sizes=[5],
        strides=[1],
        activation=activation,
        norm=None,
    )

    nbv_actor_net = CNNAndFcsNet(
        siamese_net=siamese_net,
        input_resolution=image_resolution,
        filters=[32, 64, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        norm=None,
        out_put_activation='sigmoid',
        activation=activation,
        fc_layers=[128, 64, 2], 
        low_dim_state_len=low_dim_state_len)

    nbp_actor_net = CNNAndFcsNet(
        siamese_net=siamese_net,
        input_resolution=image_resolution,
        filters=[32, 64, 64],
        kernel_sizes=[3, 3, 3],
        strides=[2, 2, 2],
        norm=None,
        activation=activation,
        fc_layers=[128, 64, 3 + 4 + 1], 
        low_dim_state_len=low_dim_state_len)

    bc_agent = BCAgent(
        nbv_actor_network=nbv_actor_net,
        nbp_actor_network=nbp_actor_net,
        camera_name=camera_name,
        lr=lr,
        weight_decay=weight_decay,
        grad_clip=grad_clip)

    return PreprocessAgent(pose_agent=bc_agent)
