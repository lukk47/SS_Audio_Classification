# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Semi-supervised learning on google command dataset with bg noise from google dataset"""


import logging
import sys
sys.path.append('./')
from experiments.run_context import RunContext
import tensorflow as tf
import numpy as np
import datasets
from mean_teacher.mean_teacher import mean_teacher
from mean_teacher import minibatching
import os

LOG = logging.getLogger('main')

flags = tf.app.flags
flags.DEFINE_integer('gpu', 0, 'GPU_number')
flags.DEFINE_integer('dataset_index', 0, 'datasets including google and urban datasets')
flags.DEFINE_integer('n_runs', 5, 'number of runs')
flags.DEFINE_integer('init_runs', 2000, 'number of runs')
FLAGS = flags.FLAGS

datasets_name = ['Audio30','Urban10']
assert FLAGS.dataset_index<= len(datasets_name), 'wrong dataset index'
data_loader = getattr(datasets, datasets_name[FLAGS.dataset_index])


def parameters():
    test_phase = True
    n_runs = FLAGS.n_runs
    n_labeled = FLAGS.n_labeled
    for n_labeled in [600, 3000, 6000, 15000, 'all']:
        for bg_noise_level in [0.1]:
            for data_seed in range(FLAGS.init_runs, FLAGS.init_runs  + n_runs):
                yield {
                    'test_phase': test_phase,
                    'n_labeled': n_labeled,
                    'data_seed': data_seed,
                    'data_type':32,
                    'bg_noise':True,
                    'bg_noise_level':bg_noise_level
                }

def run(test_phase, n_labeled, data_seed,data_type, bg_noise,bg_noise_level):

    minibatch_size = 100

    data = data_loader(n_labeled=n_labeled,
                       data_seed=data_seed,
                       test_phase=test_phase,
                       bg_noise = bg_noise)

    print('{} is loaded with {} of training samples'.format(datasets_name[FLAGS.dataset_index],data['num_train']))

    if n_labeled == 'all':
        n_labeled_per_batch =  minibatch_size
        max_consistency_cost = minibatch_size
    else:
        n_labeled_per_batch = 'vary'
        max_consistency_cost = minibatch_size* int(n_labeled) / data['num_train']

    hyper_dcit = {'input_dim': data['input_dim'],
                'label_dim': data['label_dim'],
                'cnn':'audio',
                'flip_horizontally':False,
                'max_consistency_cost': max_consistency_cost,
                'apply_consistency_to_labeled' : True,
                'adam_beta_2_during_rampup': 0.999,
                'ema_decay_during_rampup': 0.999,
                'normalize_input': True,
                'rampdown_length': 25000,
                'rampup_length': 40000,
                'training_length': 80000,
                'bg_noise':bg_noise,
                'bg_noise_input': flat(data['bg_noise_img']),
                'bg_noise_level':bg_noise_level
                }

    tf.reset_default_graph()
    runner_name = os.path.basename(__file__).split(".")[0]
    file_name = '{}_{}_{}'.format(runner_name,bg_noise_level,n_labeled)
    model = mean_teacher(RunContext(file_name, data_seed), hyper_dcit)

    training_batches = minibatching.training_batches(data.training,
                                                     minibatch_size,
                                                     n_labeled_per_batch)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(data.evaluation,
                                                                    minibatch_size)

    model.train(training_batches, evaluation_batches_fn)


def flat(x):
    tmp = np.empty((32,0))
    for row in x:
        tmp = np.concatenate([tmp,row],axis=1)
    return tmp


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    for run_params in parameters():
        run(**run_params)
