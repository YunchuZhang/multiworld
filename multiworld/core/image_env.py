import random

import cv2
import numpy as np
import warnings
from PIL import Image
from gym.spaces import Box, Dict
#from scipy.misc import imsave

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.wrapper_env import ProxyEnv
from multiworld.envs.env_util import concatenate_box_spaces
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict
import os.path as path
import pickle

import ipdb
st = ipdb.set_trace

class ImageEnv(ProxyEnv, MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
            imsize=84,
            init_camera=None,
            num_cameras=1,
            camera_space=None,
            depth=False,
            cam_info=False,
            transpose=False,
            flatten=True,
            grayscale=False,
            normalize=False,
            reward_type='wrapped_env',
            threshold=10,
            image_length=None,
            presampled_goals=None,
            non_presampled_goal_img_is_garbage=False,
            recompute_reward=False,
    ):
        """

        :param wrapped_env:
        :param imsize:
        :param init_camera:
        :param transpose:
        :param grayscale:
        :param normalize:
        :param reward_type:
        :param threshold:
        :param image_length:
        :param presampled_goals:
        :param non_presampled_goal_img_is_garbage: Set this option to True if
        you want to allow the code to work without presampled goals,
        but where the underlying env doesn't support set_to_goal. As the name,
        implies this will make it so that the goal image is garbage if you
        don't provide pre-sampled goals. The main use case is if you want to
        use an ImageEnv to pre-sample a bunch of goals.
        """
        self.quick_init(locals())
        super().__init__(wrapped_env)
        self.wrapped_env.hide_goal_markers = False
        self.imsize = imsize
        self.init_camera = init_camera
        self.num_cameras = num_cameras
        self.camera_space = camera_space
        self.depth = depth
        self.cam_info = cam_info
        self.transpose = transpose
        self.flatten = flatten
        self.grayscale = grayscale
        self.normalize = normalize
        self.recompute_reward = recompute_reward
        self.non_presampled_goal_img_is_garbage = non_presampled_goal_img_is_garbage

        if image_length is not None:
            self.image_length = image_length
        else:
            if grayscale:
                self.image_length = self.imsize * self.imsize
            else:
                self.image_length = 3 * self.imsize * self.imsize
        self.channels = 1 if grayscale else 3

        # This is torch format rather than PIL image
        self.image_shape = (self.imsize, self.imsize)
        # Flattened past image queue
        # init camera
        if init_camera is not None:
            sim = self._wrapped_env.initialize_camera(init_camera, num_cameras=self.num_cameras)

        if self.flatten:
            img_space_shape = np.array([self.num_cameras, self.image_length])
            img_space_shape = np.delete(img_space_shape, np.argwhere(img_space_shape == 1))
        else:
            img_space_shape = np.array([self.num_cameras, self.imsize, self.imsize, self.channels])
            img_space_shape = np.delete(img_space_shape, np.argwhere(img_space_shape == 1))

        if self.normalize:
            img_space = Box(0, 1, img_space_shape, dtype=np.float32)
        else:
            img_space = Box(0, 255, img_space_shape, dtype=np.uint8)

        self._img_goal = img_space.sample() #has to be done for presampling
        spaces = self.wrapped_env.observation_space.spaces.copy()
        spaces['image_observation'] = img_space
        spaces['image_desired_goal'] = img_space
        spaces['image_achieved_goal'] = img_space

        if self.depth:
            depth_space_shape = np.array([self.num_cameras, self.imsize, self.imsize])
            depth_space_shape = np.delete(depth_space_shape, np.argwhere(depth_space_shape == 1))

            depth_space = Box(0, float('inf'), depth_space_shape, dtype=np.float32)
            spaces['depth_observation'] = depth_space
            spaces['depth_desired_goal'] = depth_space


        if self.cam_info:

            # cam_angle_space_shape = (num_cameras, 2)
            # cam_angle_space = Box(float('-inf'), float('inf'), cam_angle_space_shape, dtype=np.float32)

            # cam_dist_space_shape = (num_cameras,)
            # cam_dist_space = Box(float('-inf'), float('inf'), cam_dist_space_shape, dtype=np.float32)

            # spaces['cam_angles_observation'] = cam_angle_space
            # spaces['cam_angles_goal'] = cam_angle_space

            # spaces['cam_dist_observation'] = cam_dist_space
            # spaces['cam_dist_goal'] = cam_dist_space

            # Camera information : Elevation, Azimuth, Distance, Look at point xyz
            cam_space_shape = (num_cameras, 6)
            cam_space = Box(float('-inf'), float('inf'), cam_space_shape, dtype=np.float32)
            spaces['cam_info_observation'] = cam_space
            spaces['cam_info_goal'] = cam_space

        self.return_image_proprio = False
        #TODO: Figure out what's going on here
        #if 'proprio_observation' in spaces.keys():
        #    self.return_image_proprio = True
        #    spaces['image_proprio_observation'] = concatenate_box_spaces(
        #        spaces['image_observation'],
        #        spaces['proprio_observation']
        #    )
        #    spaces['image_proprio_desired_goal'] = concatenate_box_spaces(
        #        spaces['image_desired_goal'],
        #        spaces['proprio_desired_goal']
        #    )
        #    spaces['image_proprio_achieved_goal'] = concatenate_box_spaces(
        #        spaces['image_achieved_goal'],
        #        spaces['proprio_achieved_goal']
        #    )

        self.observation_space = Dict(spaces)
        self.action_space = self.wrapped_env.action_space
        self.reward_type = reward_type
        self.threshold = threshold
        self._presampled_goals = presampled_goals
        if self._presampled_goals is None:
            self.num_goals_presampled = 0
        else:
            self.num_goals_presampled = presampled_goals[random.choice(list(presampled_goals))].shape[0]
        self._last_image = None
        # Fix initial black image problem

        # self.wrapped_env.sample_viewers(num_views=self.num_cameras)
        # self.wrapped_env.reset()
        # self._get_img()

        # Fix initial black image problem
        self.wrapped_env.sample_views(self.camera_space)
        self.wrapped_env.reset()
        self._get_img()

    def step(self, action):

        if self.camera_space:
            self.wrapped_env.sample_views(self.camera_space)

        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        if self.recompute_reward:
            # reward, done = self.compute_reward(action, new_obs)
            reward, _ = self.compute_reward(new_obs['achieved_goal'], new_obs['desired_goal'])
        self._update_info(info, obs)
        return new_obs, reward, done, info

    def _update_info(self, info, obs):
        achieved_goal = obs['image_achieved_goal']
        desired_goal = self._img_goal
        image_dist = np.linalg.norm(achieved_goal-desired_goal)
        image_success = (image_dist<self.threshold).astype(float)-1
        info['image_dist'] = image_dist
        info['image_success'] = image_success


    def reset(self):

        if self.camera_space:
            self.wrapped_env.sample_views(self.camera_space)

        obs = self.wrapped_env.reset()
        if self.num_goals_presampled > 0:
            goal = self.sample_goal()

            self._goal_rendering = goal['goal_rendering']
            goal.pop('goal_rendering', None)
            self._img_goal = goal['image_desired_goal']
            self._img_goal_depth = goal['depth_desired_goal']
            self.goal_cam_info = goal['cam_info_goal']

            self.wrapped_env.set_goal(goal)
            for key in goal:
                obs[key] = goal[key]

        elif self.non_presampled_goal_img_is_garbage:
            # This is use mainly for debugging or pre-sampling goals.
            self._img_goal, _ = self._get_img()
        else:
            env_state = self.wrapped_env.get_env_state()
            self.wrapped_env.set_to_goal(self.wrapped_env.get_goal())

            # Goal rendering is used for visualization, image goal is used
            # for learning/acting
            self._goal_rendering = self.wrapped_env.render(mode='rgb_array')
            self._img_goal, self._img_goal_depth = self._get_img()
            self.goal_cam_info = self.wrapped_env.get_camera_info()

            self.wrapped_env.set_env_state(env_state)

        return self._update_obs(obs)


    def _get_obs(self):
        return self._update_obs(self.wrapped_env._get_obs())

    def _update_obs(self, obs):
        img_obs, depths = self._get_img()
        obs['image_observation'] = img_obs

        if self.depth:
            obs['depth_observation'] = depths

        if self.cam_info:
            obs['cam_info_observation'] = self.wrapped_env.get_camera_info()

        obs['image_desired_goal'] = self._img_goal
        obs['depth_desired_goal'] = self._img_goal_depth
        obs['cam_info_goal'] = self.goal_cam_info
        obs['image_achieved_goal'] = img_obs

        if self.return_image_proprio:
            obs['image_proprio_observation'] = np.concatenate(
                (obs['image_observation'], obs['proprio_observation'])
            )
            obs['image_proprio_desired_goal'] = np.concatenate(
                (obs['image_desired_goal'], obs['proprio_desired_goal'])
            )
            obs['image_proprio_achieved_goal'] = np.concatenate(
                (obs['image_achieved_goal'], obs['proprio_achieved_goal'])
            )
        return obs


    def _get_img(self):
        if self.depth:
            image_obs, depths = self._wrapped_env.get_image(
                width=self.imsize,
                height=self.imsize,
                depth=self.depth
            )
        else:
            image_obs = self._wrapped_env.get_image(
                width=self.imsize,
                height=self.imsize,
                depth=self.depth
            )
        self._last_image = image_obs

        if self.grayscale:
            # TODO: Currently not compatible with multi-camera image_obs
            image_obs = Image.fromarray(image_obs).convert('L')
            image_obs = np.array(image_obs)

        if self.normalize:
            image_obs = image_obs / 255.0

        # Changes from (H, W, C) to (C, W, H)
        if self.transpose:
            image_obs = np.swapaxes(image_obs, -3, -1)
            if self.depth:
                depths = np.swapaxes(depths, -3, -1)

        if self.flatten:
            #TODO: currently not compatible with multi-camera image_obs
            return image_obs.flatten()

        if self.depth:
            return image_obs, depths

        return image_obs, None


    def render(self, mode='wrapped', render_goal=False):
        if mode == 'wrapped':
            self.wrapped_env.render()
        elif mode == 'rgb_array':
            if render_goal:
                return self.wrapped_env.render(mode='rgb_array'), self._goal_rendering
            else:
                return self.wrapped_env.render(mode='rgb_array')
        elif mode == 'cv2':
            # import ipdb;ipdb.set_trace()
            if self._last_image is None:
                self._last_image = self._wrapped_env.get_image(
                    width=self.imsize,
                    height=self.imsize,
                    depth=False
                )
            cv2.imshow('ImageEnv', self._last_image[4])
            cv2.waitKey(1)
        else:
            raise ValueError("Invalid render mode: {}".format(mode))

    """
    Multitask functions
    """
    def get_goal(self):
        goal = self.wrapped_env.get_goal()
        goal['image_desired_goal'] = self._img_goal
        return goal

    def set_goal(self, goal):
        ''' Assume goal contains both image_desired_goal and any goals required for wrapped envs'''
        self._img_goal = goal['image_desired_goal']
        self.wrapped_env.set_goal(goal)


    def sample_goals(self, batch_size):
        if self.num_goals_presampled > 0:
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
            return sampled_goals
        if batch_size > 1:
            warnings.warn("Sampling goal images is slow")

        goal_renderings = []
        img_goals = []
        img_goal_depths = []
        goal_cam_infos = []

        goals = self.wrapped_env.sample_goals(batch_size)
        pre_state = self.wrapped_env.get_env_state()

        for i in range(batch_size):
            goal = self.unbatchify_dict(goals, i)
            self.wrapped_env.set_to_goal(goal)

            goal_rendering = self.wrapped_env.render(mode='rgb_array')
            img_goal, img_goal_depth = self._get_img()
            goal_cam_angle = self.wrapped_env.get_camera_info()

            goal_renderings.append(goal_rendering)
            img_goals.append(img_goal)
            img_goal_depths.append(img_goal_depth)
            goal_cam_infos.append(goal_cam_info)

        self.wrapped_env.set_env_state(pre_state)
        goals['goal_rendering'] = np.array(goal_renderings)
        goals['image_desired_goal'] = np.array(img_goals)
        goals['depth_desired_goal'] = np.array(img_goal_depths)
        goals['cam_info_goal'] = np.array(goal_cam_info)
        return goals


    def compute_rewards(self, achieved_goal, desired_goal):
        achieved_goals = achieved_goal
        desired_goals = desired_goal
        # dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        if self.reward_type=='image_distance':
            return -dist
        elif self.reward_type=='image_sparse':
            return -(dist > self.threshold).astype(float)
        elif self.reward_type=='wrapped_env':
            return self.wrapped_env.compute_rewards(achieved_goals, desired_goals)
        else:
            raise NotImplementedError()

    def get_diagnostics(self, paths, **kwargs):
        statistics = self.wrapped_env.get_diagnostics(paths, **kwargs)
        for stat_name_in_paths in ["image_dist", "image_success"]:
            stats = get_stat_in_paths(paths, 'env_infos', stat_name_in_paths)
            statistics.update(create_stats_ordered_dict(
                stat_name_in_paths,
                stats,
                always_show_all_stats=True,
            ))
            final_stats = [s[-1] for s in stats]
            statistics.update(create_stats_ordered_dict(
                "Final " + stat_name_in_paths,
                final_stats,
                always_show_all_stats=True,
            ))
        return statistics

def normalize_image(image, dtype=np.float64):
    assert image.dtype == np.uint8
    return dtype(image) / 255.0

def unormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
