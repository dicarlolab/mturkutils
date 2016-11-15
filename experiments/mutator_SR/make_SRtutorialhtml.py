#!/usr/bin/env python
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
import json
import random
import math

# Todo: tutorial for canonical image version

def get_url_arrow_response_img(obj_num):
    if(obj_num == 1):
        direction = 'up'
    elif(obj_num == 2):
        direction = 'left'
    elif (obj_num == 3):
        direction = 'right'
    elif (obj_num == 4):
        direction = 'down'

    base_url = 'https://s3.amazonaws.com/mutatorsr/resources/response_images/'
    return base_url + direction + '.png'

def make(versionname, ntrials):
    if(versionname == 'hvm'):
        return makehvm(ntrials)

def makehvm(ntrials):
    #Get information to make tutorial sequence to be presented at beginning of each HIT:
    tut_combs = [('Beetle', 'Apricot_obj', 'z3', 'motoryacht')] #todo: get rid of the wonky list/tuple indexing
    meta_query = lambda x: x['var'] == 'V3'
    obj_sequencing = [0, 0, 0, 0, 1, 1, 1, 3, 2, 0] # Presentation order of objects

    assert len(obj_sequencing) == ntrials
    assert len(set(obj_sequencing)) == len(tut_combs[0])

    hvm_dataset = hvm.HvMWithDiscfade()
    tut_meta = hvm_dataset.meta
    tut_urls = hvm_dataset.publish_images(range(len(hvm_dataset.meta)), None, 'hvm_timing',
                                  dummy_upload=True)

    # Based on tut_combs, get meta indices of images to be presented
    sample_indices = []
    obj_idx_dict = {}
    for obj in tut_combs[0]:
        obj_indices = set([i for i in range(len(tut_meta)) if tut_meta[i]['obj'] == obj])
        query_indices = set([i for i in range(len(tut_meta)) if meta_query(tut_meta[i])])
        obj_idx_dict[obj] = list(obj_indices & query_indices)

    for obj_idx in obj_sequencing:
        obj= tut_combs[0][obj_idx]
        sample_indices.append(obj_idx_dict[obj].pop())

    # Create html data:
    sample_urls = []
    sample_meta = []
    for sample_idx in sample_indices:
        sample_urls.append(tut_urls[sample_idx])
        sample_meta.append({name: value for name, value in
                             zip(tut_meta[sample_idx].dtype.names, tut_meta[sample_idx].tolist())})

    response_info = [{
                        'urls': [get_url_arrow_response_img(1), get_url_arrow_response_img(2), get_url_arrow_response_img(3), get_url_arrow_response_img(4)],
                        'meta': [{'obj': obj} for obj in [o1, o2, o3, o4]],
                        'labels': [o1, o2, o3, o4]
                         }
                        for o1, o2, o3, o4 in tut_combs]

    test_urls = response_info[0]['urls']
    test_meta_entries = response_info[0]['meta']
    response_img_labels = response_info[0]['labels']


    assert len(sample_urls) == len(sample_meta) == ntrials
    assert len(test_urls) == len(test_meta_entries) == len(response_img_labels) == len(tut_combs[0])

    learningperiod_html_data = {
        'sample_urls': sample_urls, # List of strings, in order of desired image presentation
        'sample_meta': sample_meta, # List of dicts, one for each image (tabarray dict-ified)
        'tutorial_trials_per_hit': ntrials, # integer, set at top of this driver.
        'test_urls': test_urls, # List of strings, urls, one for each option.
        'test_meta_entries': test_meta_entries, # List of dicts, each tabarray dict-ified
        'response_img_labels': response_img_labels, # List of strings ['objname0', 'objname1'...]
    }

    return learningperiod_html_data
