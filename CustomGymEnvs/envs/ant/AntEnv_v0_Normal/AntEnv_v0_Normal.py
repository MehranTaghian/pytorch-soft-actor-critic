"""
modifications:
renamed env
modified filepath in __init__ method
added code to viewer_setup method to modify the camera perspective while rendering Ant
"""

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from pathlib import Path
from mujoco_py.generated import const  # do not delete; may need in viewer_setup method
import os  # modification here
from pathlib import Path

class AntEnvV1(mujoco_env.MujocoEnv, utils.EzPickle):  # modification here
    def __init__(self):
        filepath = os.path.join(Path(__file__).parent, 'xml/AntEnv_v0_Normal.xml')
        print(filepath)
        # modification here: end

        mujoco_env.MujocoEnv.__init__(self, filepath, 5)
        utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore) / self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos[7:].flat,
            self.sim.data.qvel[6:].flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

        # modification here
        self.viewer.cam.type = const.CAMERA_FIXED
        self.viewer.cam.fixedcamid = 0

        # self.viewer.cam.trackbodyid = 1
        # self.viewer.cam.distance = self.model.stat.extent * 2.0
        # self.viewer.cam.lookat[2] += .8
        # self.viewer.cam.elevation = -20