# coding=utf-8

"""
Author: angles
Date and time: 27/04/18 - 17:58
"""

import os
#
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils import create_name_experiment
from GSN import GSN

parameters = dict()
parameters['dataset'] = 'celebA_128'
parameters['train_attribute'] = '65536'
parameters['test_attribute'] = '2048_after_65536'
parameters['dim'] = 512
parameters['embedding_attribute'] = 'ScatJ4_projected{}_1norm'.format(parameters['dim'])
parameters['nb_channels_first_layer'] = 32

parameters['name_experiment'] = create_name_experiment(parameters, 'NormL1')
# print (parameters['name_experiment'])
gsn = GSN(parameters)
#gsn.train(465)
# gsn.save_originals()
gsn.generate_from_model(523)
#gsn.compute_errors(190)
# gsn.analyze_model(150)


