import numpy as np
import ipdb 
st = ipdb.set_trace
import multiworld
import gym
import cv2
import os
from xml.etree import ElementTree as et

import matplotlib as mp
# mp.use('Agg')
import matplotlib.pyplot as plt

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import init_multiple_cameras

def change_env_to_use_correct_mesh(mesh):
    path_to_xml = os.path.join('multiworld/envs/assets/sawyer_xyz/sawyer_push_box.xml')
    tree = et.parse(path_to_xml)
    root = tree.getroot()
    [x.attrib for x in root.iter('geom')][0]['mesh']=mesh

     #set the masses, inertia and friction in a plausible way

    physics_dict = {}
    physics_dict["printer"] =  ["6.0", ".00004 .00003 .00004", "1 1 .0001" ]
    physics_dict["mug1"] =  ["0.31", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001" ]
    physics_dict["mug2"] =  ["0.27", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001" ]
    physics_dict["mug3"] =  ["0.33", ".000000001 .0000000009 .0000000017", "0.008 0.008 .00001" ]
    physics_dict["can1"] =  ["0.55", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["car1"] =  ["0.2", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
    physics_dict["car2"] =  ["0.4", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
    physics_dict["car3"] =  ["0.5", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
    physics_dict["car4"] =  ["0.8", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
    physics_dict["car5"] =  ["2.0", ".0000000017 .0000000005 .0000000019", "1.2 1.2 .00001" ]
    physics_dict["boat"] =  ["7.0", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["bowl1"] =  ["0.1", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["bowl2"] =  ["0.3", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["bowl4"] =  ["0.7", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["hat1"] =  ["0.2", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]
    physics_dict["hat2"] =  ["0.4", ".00000002 .00000002 .00000001", "0.2 0.2 .0001" ]

    #set parameters
    [x.attrib for x in root.iter('inertial')][0]['mass'] = physics_dict[mesh][0]
    [x.attrib for x in root.iter('inertial')][0]['diaginertia'] = physics_dict[mesh][1]
    [x.attrib for x in root.iter('geom')][0]['friction'] = physics_dict[mesh][2]

    tree.write(path_to_xml)

H=64
W=64


def resize_image(lots_of_views_to_resize):

	shapes_ = list(lots_of_views_to_resize.shape)
	shapes_[1], shapes_[2] = H, W
	resized_images = np.zeros(shapes_)
	for i in range(lots_of_views_to_resize.shape[0]):
		resized_images[i] = cv2.resize(lots_of_views_to_resize[i], dsize=(H,W), interpolation=cv2.INTER_LINEAR)
	return resized_images
	


multiworld.register_all_envs()
change_env_to_use_correct_mesh('car1')

env = gym.make('SawyerPushAndReachEnvEasy-v0',reward_type='puck_success')
# env = ImageEnv(env,
# 		 imsize=84,
# 		 normalize=True,
# 		 init_camera=init_multiple_cameras,
# 		 num_cameras=50,
# 		 num_views=4,
# 		 depth=True,
# 		 cam_angles=True,
# 		 reward_type="wrapped_env",
# 		 flatten=False)

camera_space={'dist_low': 0.7,'dist_high': 1.5,'angle_low': 0,'angle_high': 180,'elev_low': -180,'elev_high': -90}

env = ImageEnv(
        wrapped_env=env,
        imsize=84,
        normalize=True,
        camera_space=camera_space,
        init_camera=(lambda x: init_multiple_cameras(x, camera_space)),
        num_cameras=4,
        depth=True,
        cam_info=True,
        reward_type='wrapped_env',
        flatten=False
    )


obs = env.reset()

obs_keys = obs.keys()
# import ipdb; ipdb.set_trace()
image_keys = [key for key in obs_keys if 'image' in key or 'depth' in key]
other_keys = [key for key in obs_keys if key not in image_keys]
# import ipdb; ipdb.set_trace()
image_data = {'observations.{}'.format(field_name): [resize_image(obs[field_name])] for field_name in image_keys}
other_data = {'observations.{}'.format(field_name): [obs[field_name]] for field_name in other_keys}

# obs_s = [obs['state_observation']]
# print(obs.keys())
# print(obs['image_desired_goal'].shape)
num_data = 20000

for i in range(num_data-1):
    print(i)
    a = env.action_space.sample()
    #st()
    obs, r, done, info = env.step(a)
    image_data = {'observations.{}'.format(field_name): image_data['observations.{}'.format(field_name)] + [resize_image(obs[field_name])] for field_name in image_keys}
    other_data = {'observations.{}'.format(field_name): other_data['observations.{}'.format(field_name)] + [obs[field_name]] for field_name in other_keys}
    # saving_all_data = {'observations.{}'.format(field_name): saving_all_data['observations.{}'.format(field_name)] + [obs[field_name]] for field_name in obs_keys}
    # obs_s.append(obs['state_observation'])
    # env.render()

saving_all_data = {key: np.stack(image_data[key], axis=0) for key in image_data.keys()}
saving_all_data.update({key: np.stack(other_data[key], axis=0) for key in other_data.keys()})
# import ipdb; ipdb.set_trace()

assert saving_all_data['observations.desired_goal'].shape[0] == num_data
idx = np.arange(num_data)
np.random.shuffle(idx)

train_data = {key: saving_all_data[key][idx[:int(0.5 * num_data)]] for key in saving_all_data.keys()}
val_data = {key: saving_all_data[key][idx[int(0.5 * num_data):int(0.8* num_data)]] for key in saving_all_data.keys()}
test_data = {key: saving_all_data[key][idx[int(0.8 * num_data):]] for key in saving_all_data.keys()}

np.save('push_and_reach_random_data_train.npy', train_data)
np.save('push_and_reach_random_data_val.npy', val_data)
np.save('push_and_reach_random_data_test.npy', test_data)

# np.save('push_and_reach_random_data.npy', saving_all_data)

# obs_s = np.array(obs_s)

# import ipdb; ipdb.set_trace()

# from mpl_toolkits import mplot3d
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(obs_s[:, 0], obs_s[:, 1], obs_s[:,2], c='r')
# plt.tight_layout()
# ax.set_xlabel('X-axis')
# ax.set_ylabel('Y-axis')
# ax.set_zlabel('Z-axis')
# plt.savefig('3D_heatmap.png', dpi=300)

# ## SAVING XY, YZ, ZX
# plt.figure()
# plt.scatter(obs_s[:, 0], obs_s[:, 1], c='r')
# plt.tight_layout()
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.xlim(-0.28, 0.28)
# plt.ylim(0.3, 0.9)
# plt.tight_layout()
# plt.savefig('XY_heatmap.png', dpi=300)

# plt.figure()
# plt.scatter(obs_s[:, 1], obs_s[:, 2], c='r')
# plt.tight_layout()
# plt.xlabel('Y-axis')
# plt.ylabel('Z-axis')
# plt.xlim(0.3, 0.9)
# plt.ylim(0.05, 0.3)
# plt.tight_layout()
# plt.savefig('YZ_heatmap.png', dpi=300)

# plt.figure()
# plt.scatter(obs_s[:, 2], obs_s[:, 0], c='r')
# plt.tight_layout()
# plt.xlabel('Z-axis')
# plt.ylabel('X-axis')
# plt.xlim(0.05, 0.3)
# plt.ylim(-0.28, 0.28)
# plt.tight_layout()
# plt.savefig('ZX_heatmap.png', dpi=300)


# ## SAVING FOR PUCK
# ## SAVING XY, YZ, ZX
# plt.figure()
# plt.scatter(obs_s[:, 3], obs_s[:, 4], c='r')
# plt.tight_layout()
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.xlim(-0.4, 0.4)
# plt.ylim(0.2, 1)
# plt.tight_layout()
# plt.savefig('puck_XY_heatmap.png', dpi=300)

## SAVING VIEW IMAGES
# for view in range(obs['image_desired_goal'].shape[0]):
# 	plt.figure()
# 	plt.imshow(obs['image_desired_goal'][view])
# 	plt.savefig('desired_goal_{}.png'.format(view))
# 	plt.close()

## OBS KEY DETAILS
# print([len(obs[key]) for key in obs.keys()])
# print(sum([len(obs[key]) for key in obs.keys()]))
