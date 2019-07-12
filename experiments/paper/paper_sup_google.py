# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Supervised learning of partial data on google command dataset"""


import logging
import sys
sys.path.append('./')
from experiments.run_context import RunContext
import tensorflow as tf

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
    for n_labeled in [600, 3000, 6000, 15000, 'all']:
        for data_seed in range(FLAGS.init_runs, FLAGS.init_runs + n_runs):
            yield {
                'test_phase': test_phase,
                'n_labeled': n_labeled,
                'data_seed': data_seed,
                'data_type':32,
                'bg_noise':False
            }

def run(test_phase, n_labeled, data_seed,data_type, bg_noise):

    minibatch_size = 100
    n_labeled_per_batch = minibatch_size

    data = data_loader(n_labeled=n_labeled,
                       data_seed=data_seed,
                       test_phase=test_phase)

    print('{} is loaded with {} of training samples'.format(datasets_name[FLAGS.dataset_index],data['num_train']))

    hyper_dcit = {'input_dim': data['input_dim'],
                'label_dim': data['label_dim'],
                'cnn':'audio',
                'flip_horizontally':False,
                'max_consistency_cost': 0,
                'apply_consistency_to_labeled' : False,
                'adam_beta_2_during_rampup': 0.999,
                'ema_decay_during_rampup': 0.999,
                'normalize_input': True,
                'rampdown_length': 25000,
                'rampup_length': 40000,
                'training_length': 80000
                }

    tf.reset_default_graph()
    runner_name = os.path.basename(__file__).split(".")[0]
    file_name = '{}_{}'.format(runner_name,n_labeled)
    model = mean_teacher(RunContext(file_name, data_seed), hyper_dcit)

    training_batches = minibatching.training_batches(data.training,
                                                     minibatch_size,
                                                     n_labeled_per_batch)
    evaluation_batches_fn = minibatching.evaluation_epoch_generator(data.evaluation,
                                                                    minibatch_size)

    model.train(training_batches, evaluation_batches_fn)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    for run_params in parameters():
        run(**run_params)
