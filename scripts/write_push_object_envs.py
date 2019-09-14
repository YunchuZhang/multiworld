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

def write_env_with_mesh(mesh):
    base_filename = 'sawyer_push_'
    path_to_xml = os.path.join(os.path.dirname(multiworld.__file__), 'envs/assets/sawyer_xyz/')
    tree = et.parse(path_to_xml + base_filename + 'box.xml')
    root = tree.getroot()
    [x.attrib for x in root.iter('geom')][0]['mesh']=mesh

    #set parameters
    [x.attrib for x in root.iter('inertial')][0]['mass'] = physics_dict[mesh][0]
    [x.attrib for x in root.iter('inertial')][0]['diaginertia'] = physics_dict[mesh][1]
    [x.attrib for x in root.iter('geom')][0]['friction'] = physics_dict[mesh][2]

    tree.write(path_to_xml + base_filename + mesh + '.xml')

for key in physics_dict.keys():
    write_env_with_mesh(key)
