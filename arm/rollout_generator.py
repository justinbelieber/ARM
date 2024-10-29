from multiprocessing import Value
import copy
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt

from rlbench.backend.const import TABLE_COORD

from yarr.agents.agent import Agent
from yarr.envs.env import Env
from yarr.utils.transition import ReplayTransition

from arm.ota.aux_task.aux_reward import AuxReward


class RolloutGenerator(object):
    
    def __init__(self,
                 scene_bounds,
                 cam_name,
                 device="cpu"):
        self._scene_bounds = np.array(scene_bounds)

        assert len(cam_name)==1
        self._cam_name = cam_name[0]
        
        self._device = device
        self._aux_reward = AuxReward(scene_bound=scene_bounds,
                                       voxel_size=50,
                                       device=self._device)
        self._init_obs_dict_tp0=None

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int, eval: bool):
        obs = env.reset()
        agent.reset()
        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        
        self._init_obs_dict_tp0 = copy.deepcopy(obs_history)
        self._last_obs_dict = copy.deepcopy(obs_history)
            
        for step in range(episode_length):

            prepped_data = {k:torch.tensor(np.array(v)[None], device=self._env_device) for k, v in obs_history.items()}

            act_result = agent.act(step_signal.value, prepped_data,
                                   deterministic=eval)
            gripper_target = act_result.action[:3]

            # Convert to np if not already
            agent_obs_elems = {k: np.array(v) for k, v in
                               act_result.observation_elements.items()}
            extra_replay_elements = {k: np.array(v) for k, v in
                                     act_result.replay_elements.items()}

            transition = env.step(act_result)
            obs_tp1 = dict(transition.observation)
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not transition.terminal
                if timeout:
                    transition.terminal = True
                    if "needs_reset" in transition.info:
                        transition.info["needs_reset"] = True

            obs_and_replay_elems = {}
            obs_and_replay_elems.update(obs)
            obs_and_replay_elems.update(agent_obs_elems)
            obs_and_replay_elems.update(extra_replay_elements)
            

            

            information_gain,_,roi_entropy_tp0_1, occ_ratio_tp0_1 = self._aux_reward.update_grid(target_point=gripper_target,
                                                extrinsics_tp0=obs_history["{}_camera_extrinsics".format(self._cam_name)][0],  
                                                extrinsics_tp0_1=obs_history["{}_camera_extrinsics".format(self._cam_name)][0],
                                                depth_tp0=obs_history["{}_depth".format(self._cam_name)][0],  
                                                depth_tp0_1=obs_history["{}_depth".format(self._cam_name)][0],
                                                pc_tp0=obs_history["{}_point_cloud".format(self._cam_name)][0],   
                                                pc_tp0_1=obs_history["{}_point_cloud".format(self._cam_name)][0],   
                                                roi_size=0.25)
            if eval and self._cam_name=="wrist":
                init_information_gain,roi_entropy_init,_, _ = self._aux_reward.update_grid(target_point=gripper_target,
                                                    extrinsics_tp0=self._init_obs_dict_tp0["{}_camera_extrinsics".format(self._cam_name)][0],
                                                    extrinsics_tp0_1=obs_history["{}_camera_extrinsics".format(self._cam_name)][0],
                                                    depth_tp0=self._init_obs_dict_tp0["{}_depth".format(self._cam_name)][0],
                                                    depth_tp0_1=obs_history["{}_depth".format(self._cam_name)][0],
                                                    pc_tp0=self._init_obs_dict_tp0["{}_point_cloud".format(self._cam_name)][0],
                                                    pc_tp0_1=obs_history["{}_point_cloud".format(self._cam_name)][0],
                                                    roi_size=0.25)
                transition.info["init_information_gain"] = init_information_gain/roi_entropy_init

            goal_position_interable = env.check_interaction(transition.observation["gripper_pose"][:3]+np.array(TABLE_COORD)) if eval else None
            for k in obs_history.keys():
                obs_history[k].append(transition.observation[k])
                obs_history[k].pop(0)
            #self._last_obs_dict = copy.deepcopy(obs_history)            

            transition.info["active_task_id"] = env.active_task_id

            transition.info["roi_entropy"] = roi_entropy_tp0_1
            transition.info["goal_position_interable"] = goal_position_interable

            replay_transition = ReplayTransition(
                obs_and_replay_elems, act_result.action, transition.reward,
                transition.terminal, timeout, summaries=transition.summaries,
                info=transition.info)

            if transition.terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                if len(act_result.observation_elements) > 0:
                    prepped_data = {k: torch.tensor([v], device=self._env_device) for k, v in obs_history.items()}
                    act_result = agent.act(step_signal.value, prepped_data,
                                           deterministic=eval)
                    agent_obs_elems_tp1 = {k: np.array(v) for k, v in
                                           act_result.observation_elements.items()}
                    obs_tp1.update(agent_obs_elems_tp1)
                replay_transition.final_observation = obs_tp1

            obs = dict(transition.observation)
            yield replay_transition

            if transition.info.get("needs_reset", transition.terminal):
                return
