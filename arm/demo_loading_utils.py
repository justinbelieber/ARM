import logging
from typing import List

import numpy as np
from rlbench.demo import Demo


def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


def keypoint_discovery(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):

        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or
                       last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
            episode_keypoints[-2]:
        episode_keypoints.pop(-2)
    logging.debug('Found %d keypoints.' % len(episode_keypoints),
                  episode_keypoints)
    return episode_keypoints


def keypoint_discovery_with_active_cam(demo: Demo,) -> List[List[int]]:
    episode_keypoints = []
    transition_keypoints = []
    prev_transition_index = 0
    prev_stage = 'viewpoint'
    demo_length = len(demo)
    for step, obs in enumerate(demo):
        stage = obs.stage
        #print(step,stage)
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
