import numpy as np
import cPickle as pk
import tabular as tb
import itertools
import copy
import sys

import dldata.stimulus_sets.hvm as hvm #Don't remove, or else for spooky reasons "import pymongo" doesn't work
local_library_path = '/mindhive/dicarlolab/u/mil/'
sys.path.insert(1,local_library_path)
import mldata.mldata.sets.mut as mut

from mturkutils.base import StimulusResponseExperiment
from os import path
import os
import json
import random
import math

from time import gmtime, strftime



class SR_ExperimentParams:
    # REQUIREMENTS:

    def __init__(self):
        ### Experiment metadata
        nickname_prefix = 'generic_SR_experiment' # to be used as prefix to output datafiles, and help you identify it later.

        ### Stimuli pointers

        image_urls = []  # list of image_urls to publicly available images
        meta = [] # tabarray - an associated meta structure that can be queried for each image's label, and a public URL.

        ### Task parameters:
        trials_per_hit = 200
        icon_type = 'arrows'  # 'canonical', 'circle', 'alien',
        shuffle_response_positions = False  # automatically set to False if arrows or canonical icons are used
        number_choices = 4
        labelfunc = lambda x: x['obj']  # Operates on entries of a meta structure

        ### Data to collect over population
        labelset = set(map(labelfunc, meta))
        confusions = [e for e in itertools.combinations(labelset, number_choices)]  # List of tuples, each tuple is of labels and must be of length number_choices
        overPop_effector_maps_per_confusion = math.factorial(number_choices)  # must be less than Choices, can specify
        meta_query = lambda x: x['var'] == 'V6'

        ### Specifying replicate trials
        withinHIT_reps_per_image = 1 # 1 = every trial has a unique image.
        overPop_minreps_per_image = 1  # must be geq withinHIT_reps_per_image

        ### Tutorial specifications:
        number_tutorial_trials = 10
        tutorial_content = 'HvM' # mutator, shapes

        ### Feedback
        bonus_per_correct = 0.01
        post_choice_string_feedback = True # "Correct!/Incorrect on each trial on or off?"

        ### Worker filtering
        performance_worker_filter = 'chance' # can also specify float, hard percent correct number
        # todo: one tailed t-test for leq<chance
        worker_virginities = ['img', 'comb', 'obj'] # list of meta fields for which a previous worker cannot encounter in another HIT

        self.exp_spec = {
            'nickname_prefix': nickname_prefix,
            'image_urls': image_urls,
            'meta': meta,
            'icon_type': icon_type,
            'shuffle_response_positions': shuffle_response_positions,
            'number_choices': number_choices,
            'labelfunc': labelfunc,
            'meta_query': meta_query,
            'labelset': labelset,
            'confusions': confusions,
            'overPop_effector_maps_per_confusion': overPop_effector_maps_per_confusion, # todo
            'withinHIT_reps_per_image': withinHIT_reps_per_image,
            'overPop_minreps_per_image': overPop_minreps_per_image,
            'number_tutorial_trials': number_tutorial_trials,
            'trials_per_hit': trials_per_hit,
            'tutorial_content': tutorial_content,
            'bonus_per_correct': bonus_per_correct,
            'post_choice_string_feedback': post_choice_string_feedback,
            'performance_worker_filter': performance_worker_filter,
            'worker_virginities': worker_virginities,
        }

    def save(self):
        params = self.exp_spec

        SAVEPATH = './experiment_parameter_archive/'
        fname = self.exp_spec['nickname_prefix'] + '_params_'+strftime("%Y-%m-%d--%H-%M-%S", gmtime()) + '.pk'

        savename = os.path.join(SAVEPATH, fname)
        pk.dump(params, open(savename, "wb"))
        print 'Saved the new params file at: ' + savename
        return savename

class MutatorDiscfade_SR_Experiment(SR_ExperimentParams):
    def __init__(self, labelset = None, Nway = 4):
        ### Experiment metadata
        nickname_prefix = 'mut_discfade_v6arrowSR'  # to be used as prefix to output datafiles, and help you identify it later.
        blurb = "Complete a visual object recognition task where you learn to associate novel objects with arrow keys." \
                "  We expect this HIT to take about 10 minutes, though you must finish in under 25 minutes. " \
                "By completing this HIT, you understand that you are participating in an experiment " \
                "for the Massachusetts Institute of Technology (MIT) Department of Brain and Cognitive Sciences. " \
                "You may quit at any time, and you will remain anonymous. " \
                "Contact the requester with questions or concerns about this experiment.",

        ### Stimuli pointers
        _dataset = mut.PilotAll_LargeSetWithDiscfade()
        meta = _dataset.meta
        image_urls = [self.get_image_url_from_id(n) for n in meta['id']]  # list of image_urls to publicly available images
          # tabarray - an associated meta structure that can be queried for each image's label, and a public URL.

        ### Task parameters:
        labelfunc = lambda x: x['obj']  # Operates on entries of a meta structure

        if labelset is None: 
            ### Data to collect over population
            labelset = set(map(labelfunc, meta))
            labelset = [#'algae',
                         #'ballpile',
                         #'blenderball',
                         #'blowfish',
                         #'cinnamon',
                         'city',
                         'dippindots',
                         'dirtyball',
                         'octopus',
                         #'spikybeast',
                         #'staircase',
                         #'twoshells'
                         ]
        else: 
            assert type(labelset) == list 
        
        number_choices = Nway
        
        confusions = [e for e in itertools.combinations(labelset,
                                                        number_choices)]  # List of tuples, each tuple is of labels and must be of length number_choices

        ### Tutorial specifications:

        ### Feedback
        bonus_per_correct = 0.003 # todo: right now this is specified in the .html template
        post_choice_string_feedback = True  # "Correct!/Incorrect on each trial on or off?"

        ### Worker filtering
        performance_worker_filter = 'chance'  # can also specify float, hard percent correct number
        # todo: one tailed t-test for leq<chance
        worker_virginities = ['img', 'comb', 'obj']  # todo list of meta fields for which a previous worker cannot encounter in another HIT

        self.exp_spec = {
            'nickname_prefix': nickname_prefix,
            'Amazon_description': blurb,
            'Amazon_title': 'Visual object learning - learn to categorize strange objects.',
            'image_urls': image_urls,
            'meta': meta,
            'icon_type': 'arrows',  # 'canonical', 'circle', 'alien',,
            'shuffle_response_positions': False,
            'number_choices': number_choices,
            'labelfunc_field': 'obj',
            'meta_query_tuple': ('var', 'V6'),
            'labelset': labelset,
            'confusions': confusions,
            'trials_per_hit': 200,
            'overPop_effector_maps_per_confusion': 1,
            'withinHIT_reps_per_image': 1,
            'overPop_min_reps_per_image': 20,
            'number_tutorial_trials': 10,
            'tutorial_content': 'hvm',
            'bonus_per_correct': bonus_per_correct,
            'post_choice_string_feedback': post_choice_string_feedback,
            'performance_worker_filter': performance_worker_filter,
            'worker_virginities': worker_virginities,
            'sampling_method': 'without-replacement'
        }

    def get_image_url_from_id(self, id):
        BASE_URL = 'https://s3.amazonaws.com/mutatorsr/resources/mutatorPilotDiscfade_stimuli/discfaded/'

        url = os.path.join(BASE_URL, id)
        return url


class HvM_SR_Experiment(SR_ExperimentParams):
    def __init__(self):
        ### Experiment metadata
        title = 'Visual object learning - learn to categorize objects.'
        blurb = "***Please, complete only one (1) HIT in this group. Completing more than one HIT in this " \
                "group will not lead to payment and possible blacklisting from future HITs.*** " \
                "complete a visual object recognition task where you learn to associate objects with arrow keys." \
                "We expect this HIT to take about 10 minutes, though you must finish in under 25 minutes. " \
                "By completing this HIT, you understand that you are participating in an experiment " \
                "for the Massachusetts Institute of Technology (MIT) Department of Brain and Cognitive Sciences. " \
                "You may quit at any time, and you will remain anonymous. " \
                "Contact the requester with questions or concerns about this experiment.",

        ### Stimuli pointers
        _dataset = hvm.HvMWithDiscfade()
        meta = _dataset.meta

        image_urls = _dataset.publish_images(range(len(_dataset.meta)), None, 'hvm_timing',
                                dummy_upload=True)

        ### Task parameters:
        labelfunc = lambda x: x['obj']  # Operates on entries of a meta structure

        ### Data to collect over population
        # labelset = set(map(labelfunc, meta))
        labelset = ['Apple_Fruit_obj',
                     'ELEPHANT_M',
                     'TURTLE_L',
                     '_001',
                     '_014',
                     '_18',
                     'alfa155',
                     'bear',
                     'breed_pug',
                     'f16',
                     'face0001',
                     'face0002'] # all objects in alex analysis

        labelset = [ 'bear', 'ELEPHANT_M', '_18', 'face0001', 'alfa155', 'breed_pug', 'TURTLE_L', 'Apple_Fruit_obj', 'f16', '_001'] # hvm10 for now

        labelset = ['bear', 'ELEPHANT_M', '_18', 'face0001', 'breed_pug', 'TURTLE_L', 'Apple_Fruit_obj',
                    'f16', '_001'] # hvm 10 minus alfa 155

        labelset = ['ELEPHANT_M','breed_pug', 'TURTLE_L', 'Apple_Fruit_obj',
                    'f16']  # hvm 10 minus alfa 155

        number_choices = 4
        confusions = [e for e in itertools.combinations(labelset,
                                                        number_choices)]  # List of tuples, each tuple is of labels and must be of length number_choices

        ### Feedback
        bonus_per_correct = 0.003 # todo: right now this is specified in the .html template
        post_choice_string_feedback = True  # "Correct!/Incorrect on each trial on or off?"

        ### Worker filtering
        performance_worker_filter = 'chance'  # can also specify float, hard percent correct number
        # todo: one tailed t-test for leq<chance
        worker_virginities = ['img', 'comb', 'obj']  # todo list of meta fields for which a previous worker cannot encounter in another HIT

        self.exp_spec = {
            'nickname_prefix': 'hvm_v6_arrowSR',
            'Amazon_description': blurb,
            'Amazon_title': title,
            'image_urls': image_urls,
            'meta': meta,
            'icon_type': 'arrows',  # 'canonical', 'circle', 'alien',,
            'shuffle_response_positions': False,
            'number_choices': number_choices,
            'labelfunc_field': 'obj',
            'meta_query_tuple': ('var', 'V6'),
            'labelset': labelset,
            'confusions': confusions,
            'trials_per_hit': 160,
            'overPop_effector_maps_per_confusion': 1,
            'withinHIT_reps_per_image': 1,
            'overPop_min_reps_per_image': 5,
            'number_tutorial_trials': 10,
            'tutorial_content': 'hvm',
            'bonus_per_correct': bonus_per_correct,
            'post_choice_string_feedback': post_choice_string_feedback,
            'performance_worker_filter': performance_worker_filter,
            'worker_virginities': worker_virginities,
            'sampling_method': 'with-replacement-balanced'
        }


def main(argv=[]):
    print '\nWriting new experiment params file...'

    ## Change this line to whatever class of params you have written
    params = MutatorDiscfade_SR_Experiment()
    params.save()

if __name__ == '__main__':
    main(sys.argv)
