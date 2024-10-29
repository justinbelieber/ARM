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

from arm import utils
from arm.baselines.bc_active.const import SAVE_OBS_KEYS,SAVE_OBS_ELEMENT_KEYS
from arm.ota.aux_task.aux_reward import AuxReward


class BCFRolloutGenerator(object):
    
    def __init__(self,
                 scene_bounds,
                 viewpoint_env_bounds=None,
                 device="cpu"):
        self._scene_bounds = np.array(scene_bounds)
        self._device = device
        self._viewpoint_env_bounds=np.array(viewpoint_env_bounds)
        self._init_obs_dict_tp0 = None

        # self._scene_voxels = VoxelGrid(coord_bounds=scene_bounds,
        #                     voxel_size= 100,
        #                     device=self._device,
        #                     batch_size=1,
        #                     feature_size=0,
        #                     max_num_coords=1000000,
        #                     )
        
        self._aux_reward = AuxReward(scene_bound=scene_bounds,
                                       voxel_size=50,
                                       device=self._device)

    def _get_type(self, x):
        if x.dtype == np.float64:
            return np.float32
        return x.dtype

    def act_and_execute(self,step_signal: Value, env: Env, agent: Agent,
                  timesteps: int, eval: bool,obs_dict_tp0:dict, final:bool=False,):


        prepped_data = {k:torch.tensor(np.array(v)[None], device=self._device) 
                        for k, v in obs_dict_tp0.items()}
        # nbv agent 
        nbv_output =agent.act(step=step_signal.value, 
                              observation=prepped_data,
                              layer=0, 
                              deterministic=eval)

        nbv_action = nbv_output.action.cpu().numpy()

        nbv_out_save = {'action_layer_0':nbv_action}
        
        vp_action = nbv_action * (self._viewpoint_env_bounds[4:] - self._viewpoint_env_bounds[1:3]) + self._viewpoint_env_bounds[1:3]
        
        vp_world_spher_goal = np.concatenate([self._viewpoint_env_bounds[[0]],vp_action],axis=0)
        

        vp_world_spher_goal = np.clip(vp_world_spher_goal,self._viewpoint_env_bounds[:3],self._viewpoint_env_bounds[3:])

        _,_,_,vp_world_cart_pose_goal = utils.local_spher_to_world_pose(local_viewpoint_spher_coord=vp_world_spher_goal)
        # ========================================
        if eval:
            init_vp = self._init_obs_dict_tp0["active_cam_pose"][0]
            init_vp_transition = env.step(init_vp,'vision',final)
            init_vp_obs_dict = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in init_vp_transition.observation.items()}
        # =========================================

        tp0_1_transition = env.step(vp_world_cart_pose_goal,'vision',final)
        obs_dict_tp0_1 = tp0_1_transition.observation

        obs_dict_tp0_1 = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs_dict_tp0_1.items()}

            
        # [B,T, ...]  
        prepped_data = {k:torch.tensor(np.array(v)[None], device=self._device) 
                        for k, v in obs_dict_tp0_1.items()}
        # nbp agent 
        nbp_output = agent.act(step=step_signal.value,
                               observation=prepped_data,
                               layer=1,
                               deterministic=eval)


        extra_replay_elements = {k: np.array(v) for k, v in  nbp_output.replay_elements.items()}
        

        nbp_out_save = {'action_layer_1':nbp_output.action.cpu().numpy()}


        nbp_action = nbp_output.action.cpu().numpy()
        
        gripper_target = nbp_action[:3]


        tp1_transition = env.step(nbp_action,'worker',final)
        
        terminal = tp0_1_transition.terminal or tp1_transition.terminal

        nbv_in_save= {k+'_layer_0':v[0] for k,v in obs_dict_tp0.items() if k in SAVE_OBS_KEYS} 
        nbp_in_save = {k+'_layer_1':v[0] for k,v in obs_dict_tp0_1.items() if k in SAVE_OBS_KEYS}

        obs_and_replay_elems = {}
        obs_and_replay_elems.update(nbv_in_save)
        obs_and_replay_elems.update(nbv_out_save)
        obs_and_replay_elems.update(nbp_in_save)
        obs_and_replay_elems.update(nbp_out_save)
        obs_and_replay_elems.update(extra_replay_elements)
        
        goal_position_interable = env.check_interaction(tp1_transition.observation["gripper_pose"][:3]+np.array(TABLE_COORD)) if eval else None
        ################################
        information_gain,roi_entropy_tp0,roi_entropy_tp0_1, occ_ratio_tp0_1 = self._aux_reward.update_grid(target_point=gripper_target,
                                            extrinsics_tp0=obs_dict_tp0["active_camera_extrinsics"][0],
                                            extrinsics_tp0_1=obs_dict_tp0_1["active_camera_extrinsics"][0],
                                            depth_tp0=obs_dict_tp0["active_depth"][0],
                                            depth_tp0_1=obs_dict_tp0_1["active_depth"][0],
                                            pc_tp0=obs_dict_tp0["active_point_cloud"][0],
                                            pc_tp0_1=obs_dict_tp0_1["active_point_cloud"][0],
                                            roi_size=0.25)
        if eval:
            init_information_gain,roi_entropy_init,_, _ = self._aux_reward.update_grid(target_point=gripper_target,
                                                extrinsics_tp0=init_vp_obs_dict["active_camera_extrinsics"][0],
                                                extrinsics_tp0_1=obs_dict_tp0_1["active_camera_extrinsics"][0],
                                                depth_tp0=init_vp_obs_dict["active_depth"][0],
                                                depth_tp0_1=obs_dict_tp0_1["active_depth"][0],
                                                pc_tp0=init_vp_obs_dict["active_point_cloud"][0],
                                                pc_tp0_1=obs_dict_tp0_1["active_point_cloud"][0],
                                                roi_size=0.25)
            tp1_transition.info["init_information_gain"] = init_information_gain/roi_entropy_init
            
        tp1_transition.info["roi_entropy"] = roi_entropy_tp0_1
        #tp1_transition.info["roi_reachable"] = float(roi_reachable)
        #tp1_transition.info["roi_non_empty"] = float(roi_non_empty)
        tp1_transition.info["goal_position_interable"] = goal_position_interable
        tp1_transition.info["information_gain"] = information_gain/(roi_entropy_tp0 + (1e-10))

        # s,a,s',done
        return obs_and_replay_elems, nbp_action, tp1_transition, terminal
         

    def generator(self, step_signal: Value, env: Env, agent: Agent,
                  episode_length: int, timesteps: int, eval: bool):

        obs = env.reset()
        agent.reset()

        obs_history = {k: [np.array(v, dtype=self._get_type(v))] * timesteps for k, v in obs.items()}
        self._init_obs_dict_tp0 = copy.deepcopy(obs_history)
        
        for step in range(episode_length):
            # s,a,s',terminal
            obs_and_replay_elems, tp0_1_action, tp1_transition, terminal = \
                self.act_and_execute(step_signal, env, agent, timesteps, eval,obs_history,False )            
            
            timeout = False
            if step == episode_length - 1:
                # If last transition, and not terminal, then we timed out
                timeout = not terminal
                if timeout:
                    terminal = True
                    if "needs_reset" in tp1_transition.info:
                        tp1_transition.info["needs_reset"] = True
            

            for k in obs_history.keys():
                obs_history[k].append(tp1_transition.observation[k])
                obs_history[k].pop(0)

            tp1_transition.info["active_task_id"] = env.active_task_id

            replay_transition = ReplayTransition(
                obs_and_replay_elems, tp0_1_action, tp1_transition.reward,
                terminal, timeout, summaries=tp1_transition.summaries,
                info=tp1_transition.info)


            if terminal or timeout:
                # If the agent gives us observations then we need to call act
                # one last time (i.e. acting in the terminal state).
                #debug_obs = copy.deepcopy(obs_and_replay_elems)
                
                obs_and_replay_elems, _, _, _ = self.act_and_execute(step_signal, env, agent,
                                                                     timesteps, eval,obs_history,final=True)
                #obs_tp1 = dict(tp1_transition.observation)           
                obs_and_replay_elems.pop('demo',None)

                replay_transition.final_observation = obs_and_replay_elems 


            yield replay_transition


            if tp1_transition.info.get("needs_reset", tp1_transition.terminal):
                return
