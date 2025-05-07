import gc
import logging
import os
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pickle
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig
from rlbench import CameraConfig, ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task
from rlbench.backend.utils import task_file_to_task_class
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.runners.env_runner import EnvRunner
from yarr.runners.pytorch_train_runner import PyTorchTrainRunner
from yarr.runners.pytorch_eval_runner import PyTorchEvalRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from arm import arm, c2farm, lpr, qte, ota
from arm.lpr.rollout_generator import PathArmRolloutGenerator
from arm.lpr.trajectory_action_mode import TrajectoryActionMode
from arm.ota.rollout_generator import OtaRolloutGenerator
from arm.baselines.bc_active.rollout_generator import BCFRolloutGenerator

from arm.baselines import bc, td3, dac, sac, bc_active
from arm.custom_rlbench_env import CustomRLBenchEnv
from arm.ota.custom_rlbench_env import OtaCustomRLBenchEnv

from pyrep.const import RenderMode

from arm.rollout_generator import RolloutGenerator


def _create_obs_config(camera_names: List[str], camera_resolution: List[int])->ObservationConfig:
    unused_cams = CameraConfig()
    unused_cams.set_all(False)

    used_cams = CameraConfig(

        rgb=True,
        point_cloud=True,
        mask=False,
        depth=True,
        image_size=camera_resolution,
        render_mode=RenderMode.OPENGL)

    cam_obs = []
    kwargs = {}
    for n in camera_names:
        kwargs[n] = used_cams
        cam_obs.append('%s_rgb' % n)
        cam_obs.append('%s_pointcloud' % n)

    # Some of these obs are only used for keypoint detection.
    obs_config = ObservationConfig(
        front_camera=kwargs.get('front', unused_cams),
        left_shoulder_camera=kwargs.get('left_shoulder', unused_cams),
        right_shoulder_camera=kwargs.get('right_shoulder', unused_cams),
        wrist_camera=kwargs.get('wrist', unused_cams),
        active_camera=kwargs.get('active', unused_cams),
        hmd_camera=kwargs.get('hmd', unused_cams),
        overhead_camera=kwargs.get('overhead', unused_cams),

        joint_forces=False,
        joint_positions=True,
        joint_velocities=True,
        task_low_dim_state=False,
        gripper_touch_forces=False,

        gripper_pose=True,
        gripper_open=True,
        gripper_matrix=True,
        gripper_joint_positions=True,
    )
    return obs_config


def _get_device(gpu):
    if gpu is not None and gpu >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:%d" % gpu)
        torch.backends.cudnn.enabled = torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
    return device


def eval_seed(cfg: DictConfig, env, cams, eval_device, env_device, seed) -> None:
    eval_envs = cfg.framework.eval_envs
    
    replay_ratio = None if cfg.framework.replay_ratio == 'None' else cfg.framework.replay_ratio
    replay_split = [1]
    replay_path = os.path.join(cfg.replay.path, cfg.rlbench.task, cfg.method.name, 'seed%d' % seed)
    action_min_max = None

    cwd = os.path.join(os.getcwd(),cfg.rlbench.cameras[0])
    # 
    weightsdir = os.path.join(cwd, 'seed%d' % seed, 'weights')
    logdir = os.path.join(cwd, 'seed%d' % seed) 

    rg = RolloutGenerator(scene_bounds=cfg.rlbench.scene_bounds,
                          cam_name=cfg.rlbench.cameras,
                          device=eval_device)
    
    if cfg.method.name == 'OTA':
        viewpoint_agent_bounds,viewpoint_env_bounds,viewpoint_resolution = \
            ota.launch_utils.get_viewpoint_bounds(cfg)
        
        explore_replay = ota.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            replay_path if cfg.replay.use_disk else None,
            cams, 
            env, 
            viewpoint_agent_bounds,
            viewpoint_resolution,
            cfg.method.voxel_sizes)
        
        replays = [explore_replay]

        # ota.launch_utils.fill_replay(
        #     explore_replay, 
        #     cfg.rlbench.task, 
        #     env, 
        #     cfg.rlbench.demos,
        #     cfg.method.viewpoint_augmentation,
        #     cfg.method.demo_augmentation, 
        #     cfg.method.demo_augmentation_every_n,
        #     cams, 
        #     cfg.rlbench.scene_bounds,
        #     viewpoint_agent_bounds,
        #     viewpoint_env_bounds,
        #     viewpoint_resolution,
            
        #     cfg.method.voxel_sizes, 
        #     cfg.method.bounds_offset,
        #     cfg.method.rotation_resolution, 
        #     cfg.method.crop_augmentation,
        #     cfg.method.viewpoint_align,
        #     logdir,
        #     cfg.method.reach_reward,
        #     eval_device,)
        
        agent = ota.launch_utils.create_agent(
            cfg, 
            env, 
            viewpoint_agent_bounds,
            viewpoint_env_bounds,
            viewpoint_resolution,
            cfg.rlbench.scene_bounds,
            cfg.rlbench.camera_resolution)
        
        rg = OtaRolloutGenerator(scene_bounds=cfg.rlbench.scene_bounds,
                                   viewpoint_agent_bounds=viewpoint_agent_bounds,
                                   viewpoint_resolution=viewpoint_resolution,
                                   viewpoint_env_bounds=viewpoint_env_bounds,
                                   viewpoint_align=cfg.method.viewpoint_align,
                                   reach_reward = cfg.method.reach_reward,
                                   device=eval_device)
    else:
        raise ValueError('Method %s does not exists.' % cfg.method.name)
    
    wrapped_replays = [PyTorchReplayBuffer(r) for r in replays]
    stat_accum = SimpleAccumulator(eval_video_fps=30)
    
    if action_min_max is not None:
        os.makedirs(logdir, exist_ok=True)
        with open(os.path.join(logdir, 'action_min_max.pkl'), 'wb') as f:
            pickle.dump(action_min_max, f)

    env_runner = EnvRunner(
        train_env=env, agent=agent,
        train_replay_buffer=explore_replay,
        num_train_envs=0,
        num_eval_envs=eval_envs,
        episodes=2, # 可以评测2个episode
        episode_length=cfg.rlbench.episode_length, # 每个episode的长度
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        env_device=env_device,
        rollout_generator=rg)
    
    eval_runner = PyTorchEvalRunner(
        agent, env_runner,
        wrapped_replays, eval_device, stat_accum,

        iterations=cfg.framework.training_iterations,
        save_freq=cfg.framework.save_freq,
        log_freq=cfg.framework.log_freq,
        logdir=logdir,
        weightsdir=weightsdir,
        replay_ratio=replay_ratio,
        transitions_before_train=cfg.framework.transitions_before_train,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging)
        

    print("==================Before eval_runner.start()==================")    
    eval_runner.start()
    del eval_runner
    print("==================After eval_runner.start()==================") 

    # train_runner = PyTorchTrainRunner(
    #     agent, env_runner,
    #     wrapped_replays, eval_device, replay_split, stat_accum,

    #     iterations=cfg.framework.training_iterations,
    #     save_freq=cfg.framework.save_freq,
    #     log_freq=cfg.framework.log_freq,
    #     logdir=logdir,
    #     weightsdir=weightsdir,
    #     replay_ratio=replay_ratio,
    #     transitions_before_train=cfg.framework.transitions_before_train,
    #     tensorboard_logging=cfg.framework.tensorboard_logging,
    #     csv_logging=cfg.framework.csv_logging)
    
    # train_runner.start()
    # del train_runner
    
    
    del env_runner
    del agent
    del env
    gc.collect()
    torch.cuda.empty_cache()
    


@hydra.main(config_name='config', config_path='conf')
def main(cfg: DictConfig) -> None:
    logging.info('\n' + OmegaConf.to_yaml(cfg))

    eval_device = _get_device(cfg.framework.gpu)
    env_device = _get_device(cfg.framework.env_gpu)
    logging.info('Using eval device %s.' % str(eval_device))
    logging.info('Using env device %s.' % str(env_device))

    gripper_mode = Discrete()
    if cfg.method.name == 'PathARM':
        arm_action_mode = TrajectoryActionMode(cfg.method.trajectory_points)
    else:
        arm_action_mode = EndEffectorPoseViaPlanning(collision_checking=False)
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [t.replace('.py', '') for t in os.listdir(task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    if cfg.rlbench.task not in task_files:
        raise ValueError('Task %s not recognised!.' % cfg.rlbench.task)

    task_class = task_file_to_task_class(cfg.rlbench.task)

    cfg.rlbench.cameras = cfg.rlbench.cameras if isinstance(
        cfg.rlbench.cameras, ListConfig) else [cfg.rlbench.cameras]
    obs_config = _create_obs_config(cfg.rlbench.cameras,
                                    cfg.rlbench.camera_resolution)
    

    assert len(cfg.rlbench.cameras) == 1
    cwd = os.path.join(os.getcwd(),cfg.rlbench.cameras[0])
    if not os.path.exists(cwd):
        os.mkdir(cwd) 
    
    logging.info('CWD:' + cwd)
    existing_seeds = len(list(filter(lambda x: 'seed' in x, os.listdir(cwd))))

    viewpoint_agent_bounds,viewpoint_env_bounds,viewpoint_resolution = \
        ota.launch_utils.get_viewpoint_bounds(cfg)


    env = OtaCustomRLBenchEnv(
        task_class=task_class, 
        observation_config=obs_config,
        viewpoint_env_bounds=viewpoint_env_bounds,
        viewpoint_agent_bounds=viewpoint_agent_bounds,
        viewpoint_resolution=viewpoint_resolution,
        action_mode=action_mode, 
        dataset_root=cfg.rlbench.demo_path,
        episode_length=cfg.rlbench.episode_length,
        headless=False,
        floating_cam=cfg.method.floating_cam if "floating_cam" in cfg.method.keys() else True,
        robot_setup=cfg.method.robot,
        time_in_state=cfg.method.time_in_state,
        low_dim_size=cfg.method.low_dim_size if "low_dim_size" in cfg.method.keys() else None)

        
    if len(cfg.framework.setseeds) == 0:
        for seed in range(existing_seeds, existing_seeds + cfg.framework.seeds):
            logging.info('Starting seed %d.' % seed)
            # run_seed(cfg, env, cfg.rlbench.cameras, train_device, env_device, seed)
            eval_seed(cfg, env, cfg.rlbench.cameras, eval_device, env_device, 0)
    else:
        for seed in cfg.framework.setseeds:
            if seed not in list(filter(lambda x: 'seed' in x, os.listdir(cwd))):
                logging.info('Starting seed %d.' % seed)
                #  run_seed(cfg, env, cfg.rlbench.cameras, train_device, env_device, seed)
                eval_seed(cfg, env, cfg.rlbench.cameras, eval_device, env_device, seed)

                break
            else:
                logging.info('Seed %d Done.' % seed)


if __name__ == '__main__':
    main()
