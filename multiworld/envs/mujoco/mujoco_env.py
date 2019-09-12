import os
import random

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class MujocoEnv(gym.Env):
    """
    This is a simplified version of the gym MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """
    def __init__(self,
                 model_path,
                 frame_skip,
                 device_id=-1,
                 automatically_set_spaces=False):

        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None

        self.num_cameras = None
        self.viewers = []

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        if device_id == -1 and 'gpu_id' in os.environ:
            device_id =int(os.environ['gpu_id'])
        self.device_id = device_id
        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()
        if automatically_set_spaces:
            observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
            assert not done
            self.obs_dim = observation.size

            bounds = self.model.actuator_ctrlrange.copy()
            low = bounds[:, 0]
            high = bounds[:, 1]
            self.action_space = spaces.Box(low=low, high=high)

            high = np.inf*np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if n_frames is None:
            n_frames = self.frame_skip
        if self.sim.data.ctrl is not None and ctrl is not None:
            self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            # window size used for old mujoco-py:
            width, height = 500, 500
            #width, height = 4000, 4000
            viewer = self._get_viewer(mode='rgb_array')
            viewer.render(width, height, camera_id=None)
            data = viewer.read_pixels(width, height, depth=False)

            # we set self.viewer as None as a hack to deal with rendering getting
            # messed up by get_image
            self.viewer = None

            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def close(self):
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

    def _get_viewer(self, mode='human'):
        if self.viewer is None:
            if mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=self.device_id)
            else:
                self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])


    def sample_views(self, cam_space):
        dists = np.random.uniform(cam_space['dist_low'], cam_space['dist_high'], self.num_cameras)
        angles = np.random.uniform(cam_space['angle_low'], cam_space['angle_high'], self.num_cameras)
        elevs = np.random.uniform(cam_space['elev_low'], cam_space['elev_high'], self.num_cameras)

        for i, viewer in enumerate(self.viewers):
            viewer.cam.trackbodyid = 0
            viewer.cam.distance = dists[i]
            viewer.cam.azimuth = angles[i]
            viewer.cam.elevation = elevs[i]
            viewer.cam.trackbodyid = -1


    def get_image(self, width=84, height=84, camera_name=None, depth=False):
        if self.num_cameras == 1:
            return self.sim.render(
                width=width,
                height=height,
                camera_name=camera_name,
            )

        images = []
        if depth:
            depths = []

        for viewer in self.viewers:
            # TODO handle camera_name to get camera_id

            # This is a hack to make sure the correct image is
            # read for every viewer
            viewer.read_pixels(width, height, depth=True)
            viewer.render(width=width, height=height, camera_id=None)
            if depth:
                im, d = viewer.read_pixels(width, height, depth=True)
                images.append(im.copy())

                near = viewer.scn.camera[0].frustum_near
                far = viewer.scn.camera[0].frustum_far
                d = far * near / (far - (far - near) * d)
                d = d/2
                #d[d > 5] = 5.0
                depths.append(d.copy())
            else:
                im = viewer.read_pixels(width, height, depth=False)
                images.append(im.copy())

        if depth:
            return np.array(images), np.array(depths)

        return np.array(images)


    def get_camera_angles(self):
        angles = []
        for viewer in self.viewers:
            angles.append([viewer.cam.elevation,
                           viewer.cam.azimuth])

        return np.array(angles)


    def get_camera_distances(self):
        return np.array([viewer.cam.distance for viewer in self.viewers])


    def get_camera_info(self):
        return np.array([[viewer.cam.elevation,
                          viewer.cam.azimuth,
                          viewer.cam.distance,
                          viewer.cam.lookat[0],
                          viewer.cam.lookat[1],
                          viewer.cam.lookat[2]] for viewer in self.viewers])


    def initialize_camera(self, init_fctn, num_cameras=1):
        self.num_cameras = num_cameras
        sim = self.sim
        cameras = []
        for i in range(self.num_cameras):
            viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=self.device_id)
            self.viewers.append(viewer)
            cameras.append(viewer.cam)
        # viewer = mujoco_py.MjViewer(sim)
        if self.num_cameras == 1:
            init_fctn(cameras[0])
            sim.add_render_context(self.viewers[0])
        else:
            init_fctn(cameras)
