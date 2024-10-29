import numpy as np
from pyrep.const import RenderMode
from pyrep.objects import Dummy, VisionSensor
from tools.cinematic_recorder import CircleCameraMotion


class RLBenchCinematic(object):

    def __init__(self,env):
        cam_placeholder = Dummy('cam_cinematic_placeholder1')
        self._cam_base = Dummy('cam_cinematic_base')
        self._cam = VisionSensor.create([1080, 960])
        self._cam.set_explicit_handling(True)
        self._cam.set_pose(cam_placeholder.get_pose()+np.array([0.3,0.3,0.0,0.0,0.0,0.0,0.0]))
        self._cam.set_parent(cam_placeholder)
        self._cam.set_render_mode(RenderMode.OPENGL3)
        self._frames = []
        self._obs = []
        self._cam_motion = CircleCameraMotion(self._cam, self._cam_base, 0.005)
        self._cam_motion.save_pose()
        self._env = env

    def callback(self):
        self._cam.handle_explicitly()
        cap = (self._cam.capture_rgb() * 255).astype(np.uint8)
        self._frames.append(cap)
        self._cam_motion.step()
        self._obs.append(self._env._task._scene.get_observation())
        
    def reset(self):
        self._cam_motion.restore_pose()
        
    

    def empty(self):
        self._frames.clear()
        self._obs.clear()

    @property
    def frames(self):
        return list(self._frames)
    @property
    def observations(self):
        return list(self._obs)
