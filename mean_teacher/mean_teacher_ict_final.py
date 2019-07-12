# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"Mean teacher model"

import logging
import os
from collections import namedtuple

import tensorflow as tf
from tensorflow.contrib import metrics, slim
from tensorflow.contrib.metrics import streaming_mean

from . import nn
from .framework import assert_shape
from . import string_utils
from .loss import errors, classification_costs, consistency_costs, total_costs
from . import model
from .ramp import ramp_value
from . import ict
LOG = logging.getLogger('main')

import numpy as np

class mean_teacher:
    hyper = {

        # architecture of cnn
        'cnn': 'tower',
        'input_dim': (32,32,3),
        'label_dim':(),
        'sig':False,

        # Consistency hyperparameters
        'ema_consistency': True,
        'apply_consistency_to_labeled': True,
        'max_consistency_cost': 100.0,
        'ema_decay_during_rampup': 0.99,
        'ema_decay_after_rampup': 0.999,
        'consistency_trust': 0.0,

        # Optimizer hyperparameters
        'max_learning_rate': 0.003,
        'adam_beta_1_before_rampdown': 0.9,
        'adam_beta_1_after_rampdown': 0.5,
        'adam_beta_2_during_rampup': 0.99,
        'adam_beta_2_after_rampup': 0.999,
        'adam_epsilon': 1e-8,

        # Architecture hyperparameters
        'input_noise': 0.15,
        'student_dropout_probability': 0.5,
        'teacher_dropout_probability': 0.5,

        # Training schedule
        'rampup_length': 40000,
        'rampdown_length': 25000,
        'training_length': 150000,

        # Input augmentation
        'flip_horizontally': False,
        'translate': True,
        

        # Whether to scale each input image to mean=0 and std=1 per channel
        # Use False if input is already normalized in some other way
        'normalize_input': True,

        # Output schedule
        'print_span': 20,
        'evaluation_span': 500,

        # specially designed for audio dataset
        'bg_noise':False,
        'bg_noise_input': None,
        'bg_noise_level': 0,


        'optimizer': 'adam',
        'ict': False,
        'mixup': 'interpolation',
        'cons_loss':'logits'
    }

    #pylint: disable=too-many-instance-attributes
    def __init__(self, run_context=None, hyper_dict={}):
        
        #inilization of hyper
        for i in hyper_dict:
            assert i in self.hyper, "Wrong hyper dict '{}'!".format(i)
            self.hyper[i] = hyper_dict[i]

        if self.hyper['bg_noise']:
            self.bg_noise_input = tf.convert_to_tensor(self.hyper['bg_noise_input'],dtype=tf.float32)
        else:
            self.bg_noise_input = tf.convert_to_tensor(np.zeros((32,32)),dtype=tf.float32)
        # inilization model
        print('{} is initliazed!'.format(self.hyper['cnn']))
        self.cnn = getattr(model,self.hyper['cnn'])

        if run_context is not None:
            self.training_log = run_context.create_train_log('training')
            self.validation_log = run_context.create_train_log('validation')
            self.checkpoint_path = os.path.join(run_context.transient_dir, 'checkpoint')
            self.tensorboard_path = os.path.join(run_context.result_dir, 'tensorboard')

        with tf.name_scope("placeholders"):
            # self.images = tf.placeholder(dtype=tf.float32, shape=(None,) + self.hyper['input_dim'], name='images')
            self.inputs_1 = tf.placeholder(dtype=tf.float32, shape=(None,) + self.hyper['input_dim'], name='inputs_1')
            self.inputs_2 = tf.placeholder(dtype=tf.float32, shape=(None,) + self.hyper['input_dim'], name='inputs_2')
            self.inputs_1_un = tf.placeholder(dtype=tf.float32, shape=(None,) + self.hyper['input_dim'], name='inputs_1_un')
            self.inputs_2_un = tf.placeholder(dtype=tf.float32, shape=(None,) + self.hyper['input_dim'], name='inputs_2_un')
            self.labels_1 = tf.placeholder(dtype=tf.int32, shape=(None,) + self.hyper['label_dim'], name='labels_1')
            self.labels_2 = tf.placeholder(dtype=tf.int32, shape=(None,) + self.hyper['label_dim'], name='labels_2')
            # self.images_un = tf.placeholder(dtype=tf.float32, shape=(None,) + self.hyper['input_dim'], name='images_un')

            # self.lam[0] for samples with label self.lam[1] for samples without label
            self.lam = tf.placeholder(dtype=tf.float32, shape=(2), name='lambda')
            self.is_training = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')

        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        tf.add_to_collection("init_in_init", self.global_step)

        with tf.name_scope("ramps"):
            self.learning_rate, self.cons_coefficient, \
            self.adam_beta_1, self.adam_beta_2, \
            self.ema_decay = ramp_value(self.global_step,self.hyper)


        training_data = {
            'inputs_1':self.inputs_1,
            'inputs_2':self.inputs_2,
            'labels_1':self.labels_1,
            'labels_2':self.labels_2,
            'inputs_1_un':self.inputs_1_un,
            'inputs_2_un':self.inputs_2_un,
            'lam':self.lam
        }


        (   self.class_logits_1,
            self.class_logits_mixed,
            self.class_logits_mixed_un,
            self.class_logits_ema,
            self.class_logits_ema_un
        ) = self.inference_ict(
            training_data,
            is_training=self.is_training,
            ema_decay=self.ema_decay,
            input_noise=self.hyper['input_noise'],
            student_dropout_probability=self.hyper['student_dropout_probability'],
            teacher_dropout_probability=self.hyper['teacher_dropout_probability'],
            normalize_input=self.hyper['normalize_input'],
            flip_horizontally=self.hyper['flip_horizontally'],
            translate=self.hyper['translate'])


        with tf.name_scope("objectives"):
            self.mean_error_1, self.errors_1 = errors(self.class_logits_1, self.labels_1,sig = self.hyper['sig'])
            self.mean_error_ema, self.errors_ema = errors(self.class_logits_ema, self.labels_1,sig = self.hyper['sig'])

            self.mean_class_cost_1, self.class_costs_1 = classification_costs(
                self.class_logits_1, self.labels_1,sig = self.hyper['sig'])
            self.mean_class_cost_ema, self.class_costs_ema = classification_costs(
                self.class_logits_ema, self.labels_1,sig = self.hyper['sig'])
            if self.hyper['mixup']=='mixup':
                # import pdb;pdb.set_trace()
                self.labels_mixed = tf.one_hot(self.labels_1,depth= 30) * self.lam[0] + \
                    tf.one_hot(self.labels_2,depth= 30) * (1-self.lam[0])
                self.mean_class_cost_mixed, self.class_costs_mixed = classification_costs(
                self.class_logits_mixed, self.labels_mixed,sig = 'softmax not sparse')
                self.class_costs = self.class_costs_mixed

            elif self.hyper['mixup']=='interpolation':
                self.mean_class_cost_mixed_1, self.class_costs_mixed_1 = classification_costs(
                    self.class_logits_mixed, self.labels_1,sig = self.hyper['sig'])
                self.mean_class_cost_mixed_2, self.class_costs_mixed_2 = classification_costs(
                    self.class_logits_mixed, self.labels_2,sig = self.hyper['sig'])
                self.class_costs = self.class_costs_mixed_1 * self.lam[0] +  self.class_costs_mixed_2 * (1-self.lam[0])
            else:
                assert False, 'Wrong mixup type input!'

            # labeled_consistency = self.hyper['apply_consistency_to_labeled']
            # consistency_mask = tf.logical_or(tf.equal(self.labels, -1), labeled_consistency)
            consistency_mask = 1
            self.mean_cons_cost_mt, self.cons_costs_mt = consistency_costs(
                self.class_logits_mixed_un, self.class_logits_ema_un, 
                self.cons_coefficient, consistency_mask, self.hyper['consistency_trust'],
                sig = self.hyper['cons_loss'])

            self.mean_total_cost_mt, self.total_costs_mt = total_costs(
                self.class_costs, self.cons_costs_mt)

            self.cost_to_be_minimized = self.mean_total_cost_mt


        with tf.name_scope("train_step"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if self.hyper['optimizer']=='adam':
                    self.train_step_op = nn.adam_optimizer(self.cost_to_be_minimized,
                                                       self.global_step,
                                                       learning_rate=self.learning_rate,
                                                       beta1=self.adam_beta_1,
                                                       beta2=self.adam_beta_2,
                                                       epsilon=self.hyper['adam_epsilon'])
                elif self.hyper['optimizer']=='sgd':
                    self.train_step_op = nn.sgd_optimizer(self.cost_to_be_minimized,
                                                       self.global_step,
                                                       learning_rate=self.hyper['max_learning_rate'])
                else:
                    assert False, 'Wrong optimizer input!'


        self.training_metrics = {
            "learning_rate": self.learning_rate,
            "adam_beta_1": self.adam_beta_1,
            "adam_beta_2": self.adam_beta_2,
            "ema_decay": self.ema_decay,
            "cons_coefficient": self.cons_coefficient,
            "train/error/1": self.mean_error_1,
            "train/error/ema": self.mean_error_ema,
            "train/class_cost/1": self.mean_class_cost_1,
            "train/class_cost/ema": self.mean_class_cost_ema,
            "train/cons_cost/mt": self.mean_cons_cost_mt,
            "train/total_cost/mt": self.mean_total_cost_mt,
        }

        with tf.variable_scope("validation_metrics") as metrics_scope:
            self.metric_values, self.metric_update_ops = metrics.aggregate_metric_map({
                "eval/error/1": streaming_mean(self.errors_1),
                "eval/error/ema": streaming_mean(self.errors_ema),
                "eval/class_cost/1": streaming_mean(self.class_costs_1),
                "eval/class_cost/ema": streaming_mean(self.class_costs_ema),
            })
            metric_variables = slim.get_local_variables(scope=metrics_scope.name)
            self.metric_init_op = tf.variables_initializer(metric_variables)

        self.result_formatter = string_utils.DictFormatter(
            order=["eval/error/ema", "error/1", "class_cost/1", "cons_cost/mt"],
            default_format='{name}: {value:>10.6f}',
            separator=",  ")
        self.result_formatter.add_format('error', '{name}: {value:>6.1%}')

        with tf.name_scope("initializers"):
            init_init_variables = tf.get_collection("init_in_init")
            train_init_variables = [
                var for var in tf.global_variables() if var not in init_init_variables
            ]
            self.init_init_op = tf.variables_initializer(init_init_variables)
            self.train_init_op = tf.variables_initializer(train_init_variables)

        self.saver = tf.train.Saver()
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)
        self.run(self.init_init_op)
        self.save_tensorboard_graph()

    def __getitem__(self, key):
        return self.hyper[key]

    def __setitem__(self, key, value):
        self.hyper[key]=value

    def batch_extract(self,batch):

        batch_labeled = batch[batch['y']!=-1]
        batch_unlabeled = batch[batch['y']==-1]

        input_a, input_b, target_a, target_b, lam_lab = ict.mixup_data_sup(batch_labeled['x'],batch_labeled['y'])
        input_a_un, input_b_un,_,_,lam_un =  ict.mixup_data_sup(batch_unlabeled['x'],batch_unlabeled['y'])
        
        batch_new={'inputs_1': input_a,
            'inputs_2': input_b,
            'inputs_1_un': input_a_un,
            'inputs_2_un': input_b_un,
            'target_a': target_a,
            'target_b' : target_b,
            'lam': (lam_lab,lam_un)
            }

        return batch_new

    def batch_extract_valid(self,batch):

        batch_new={'inputs_1': batch['x'],
            'inputs_2':batch['x'],
            'inputs_1_un': batch['x'],
            'inputs_2_un': batch['x'],
            'target_a': batch['y'],
            'target_b' : batch['y'],
            'lam': (0,0)
            }
        return batch_new

    def train(self, training_batches, evaluation_batches_fn):
        batch = next(training_batches)
        batch_new = self.batch_extract(batch)
        self.run(self.train_init_op, self.feed_dict(batch_new))
        LOG.info("Model variables initialized")
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()
        for batch in training_batches:
            batch_new = self.batch_extract(batch)
            
            results, step, _ = self.run([self.training_metrics, self.global_step, self.train_step_op],
                                  self.feed_dict(batch_new))

            # self.training_log.record(step, {**results })

            if step % self.hyper['print_span']==0: 
                LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))
            if step > self.hyper['training_length']:
                break
            if step % self.hyper['evaluation_span'] ==0 and step!=0:
                self.evaluate(evaluation_batches_fn)
                self.save_checkpoint()
        self.evaluate(evaluation_batches_fn)
        self.save_checkpoint()

    def evaluate(self, evaluation_batches_fn):
        self.run(self.metric_init_op)
        for batch in evaluation_batches_fn():
            batch_new = self.batch_extract_valid(batch)
            self.run(self.metric_update_ops, feed_dict=self.feed_dict(batch_new,is_training=False))
        step = self.run(self.global_step)
        results = self.run(self.metric_values)
        self.validation_log.record(step, results)
        LOG.info("step %5d:   %s", step, self.result_formatter.format_dict(results))


    def run(self, *args, **kwargs):
        return self.session.run(*args, **kwargs)

    def feed_dict(self, batch, is_training=True):
        return {
            self.inputs_1: batch['inputs_1'],
            self.inputs_2: batch['inputs_2'],
            self.inputs_1_un: batch['inputs_1_un'],
            self.inputs_2_un: batch['inputs_2_un'],
            self.labels_1: batch['target_a'],
            self.labels_2: batch['target_b'],
            self.lam: batch['lam'],
            self.is_training: is_training
        }


    def save_checkpoint(self):
        path = self.saver.save(self.session, self.checkpoint_path, global_step=self.global_step)
        LOG.info("Saved checkpoint: %r", path)

    def save_tensorboard_graph(self):
        writer = tf.summary.FileWriter(self.tensorboard_path)
        writer.add_graph(self.session.graph)
        LOG.info("Saved tensorboard graph to %r", writer.get_logdir())

    def inference_ict(self,inputs, is_training, ema_decay, input_noise, student_dropout_probability, teacher_dropout_probability,
                  normalize_input, flip_horizontally, translate):

        tower_args = dict(is_training=is_training,
                          input_noise=input_noise,
                          normalize_input=normalize_input,
                          flip_horizontally=flip_horizontally,
                          translate=translate,
                          bg_noise = self.hyper['bg_noise'],
                          bg_noise_level =self.hyper['bg_noise_level'],
                          bg_noise_input =self.bg_noise_input )


        data = inputs['lam'][0]*inputs['inputs_1']+ (1 - inputs['lam'][0])*inputs['inputs_2']
        data_un = inputs['lam'][1]*inputs['inputs_1_un']+ (1 - inputs['lam'][1])*inputs['inputs_2_un']
        with tf.variable_scope("initialization") as var_scope:
            _ = self.cnn(inputs=data,**tower_args, dropout_probability=student_dropout_probability, is_initialization=True)
        with tf.variable_scope(var_scope, reuse=True) as var_scope:
            class_logits_mixed = self.cnn(inputs = data,**tower_args, dropout_probability=student_dropout_probability)
        with tf.variable_scope(var_scope, reuse=True) as var_scope:
            class_logits_1 = self.cnn(inputs = inputs['inputs_1'],**tower_args, dropout_probability=student_dropout_probability)
        with tf.variable_scope(var_scope, reuse=True) as var_scope:
            class_logits_mixed_un = self.cnn(inputs = data_un,**tower_args, dropout_probability=student_dropout_probability)

        # ema
        model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay)

        with tf.control_dependencies(update_ops):
            ema_op = ema.apply(model_vars)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)

        def ema_getter(getter, name, *args, **kwargs):
            var = getter(name, *args, **kwargs)
            if 'moving_mean' in var.name or 'moving_variance' in var.name:
                return var
            else:
                assert var in model_vars, "Unknown variable {}.".format(var)
                return ema.average(var)

        with tf.variable_scope(var_scope, custom_getter = ema_getter, reuse = True) as var_scope:
            class_logits_ema = self.cnn(inputs = inputs['inputs_1'], **tower_args, dropout_probability=teacher_dropout_probability)
            class_logits_ema = tf.stop_gradient(class_logits_ema)
        with tf.variable_scope(var_scope, reuse=True) as var_scope:
            class_logits_ema_1_un = self.cnn(inputs = inputs['inputs_1_un'], **tower_args, dropout_probability=teacher_dropout_probability)
            class_logits_ema_1_un = tf.stop_gradient(class_logits_ema_1_un)
        with tf.variable_scope(var_scope, reuse=True) as var_scope:
            class_logits_ema_2_un = self.cnn(inputs = inputs['inputs_2_un'], **tower_args, dropout_probability=teacher_dropout_probability)
            class_logits_ema_2_un = tf.stop_gradient(class_logits_ema_2_un)
        
        
        class_logits_ema_un = inputs['lam'][1]*class_logits_ema_1_un + (1 - inputs['lam'][1])*class_logits_ema_2_un

        return (class_logits_1,class_logits_mixed,class_logits_mixed_un,class_logits_ema,class_logits_ema_un)
