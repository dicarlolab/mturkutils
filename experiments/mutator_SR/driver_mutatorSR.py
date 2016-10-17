#!/usr/bin/env python
import numpy as np
import cPickle as pk
import tabular as tb
import itertools
import copy
import sys
import dldata.stimulus_sets.hvm as hvm #Don't remove, or else for spooky reasons "import pymongo" doesn't work
from mturkutils.base import StimulusResponseExperiment
from os import path
import json
import random
import math

#SELECTED_BASIC_OBJS = ['octopus', 'dippindots', 'blenderball', 'dirtyball']
#REPEATS_PER_QE_IMG = 0 # Number of times to repeat selected images, specified below, for "Quality Estimation" (i1 internal consistency) purposes

TRIALS_PER_HIT = 200 # Number of experimental trials, without tutorial trials.
tutorial_trials_per_hit = 0  # Number of learning trials
num_HITs_per_confusion = 1
UNIQUE_WORKERS_PER_HIT = 15
USE_CANONICAL_IMAGE_RESPONSES = True
abstract_images = False
# Todo: tutorial for canonical image version


def specify_SRexperiment_parameters():
    # REQUIREMENTS:

    ### Experiment metadata
    nickname_prefix = 'mutatorSR_var6_arrow_responses' # to be used as prefix to output datafiles, and help you identify it later.

    ### Stimuli pointers
    image_urls = []  # list of image_urls to publicly available images
    meta = [] # tabarray - an associated meta structure that can be queried for each image's label, and a public URL.

    ### Task parameters:
    icon_type = 'arrows'  # 'canonical', 'circle', 'alien',
    shuffle_response_positions = False  # automatically set to False if arrows or canonical icons are used
    number_choices = 4
    labelfunc = lambda x: x['obj']  # Operates on entries of a meta structure

    ### Data to collect over population
    _labelset = set(map(labelfunc, meta))
    confusions = [e for e in itertools.combinations(_labelset, number_choices)]  # List of tuples, each tuple is of labels and must be of length number_choices
    overPop_effector_maps_per_confusion = math.factorial(number_choices)  # must be less than Choices, can specify

    ### Specifying replicate trials
    withinHIT_reps_per_image = 1 # 1 = every trial has a unique image.
    overPop_reps_per_image = 1  # must be geq withinHIT_reps_per_image

    ### Tutorial specifications:
    number_tutorial_trials = 10
    tutorial_content = 'HvM' # mutator, shapes

    ### Feedback
    bonus_per_correct = 0.01
    post_choice_string_feedback = True # "Correct!/Incorrect on each trial on or off?"

    ### Worker filtering
    performance_filter = 'chance' # can also specify float, hard percent correct number
    # todo: one tailed t-test for leq<chance
    worker_virginities = ['img', 'comb', 'obj'] # list of meta fields for which a previous worker cannot encounter in another HIT




# Todo: add a canvas for these. The solution right now is hacky - the response_image_indices for these stay the same (1, 2, 3, 4) but the labels are shuffled for every HIT.
def get_url_labeled_resp_img(obj, obj_num):
    if(obj_num == 1):
        direction = 'up'
    elif(obj_num == 2):
        direction = 'left'
    elif (obj_num == 3):
        direction = 'right'
    elif (obj_num == 4):
        direction = 'down'

    if USE_CANONICAL_IMAGE_RESPONSES:
        base_url = 'https://s3.amazonaws.com/mutatorsr/resources/response_images/'
        if(abstract_images == True):
            return base_url + 'abstract' + str(obj_num) + '.png'

        return base_url + 'pilot11_canonical/_gray_'+obj + '.png'


    base_url = 'https://s3.amazonaws.com/mutatorsr/resources/response_images/'
    return base_url + direction + '.png'


def get_url_labeled_resp_img_hardcoded(cat):
    # Get response image for this 'category' cat.
    # Define directions (hardcode for now).
    # TODO: randomize objects and directions, over HITs. 1 HIT/worker
    if(cat == 'octopus'):
        direction = 'up'
    elif(cat == 'dippindots'):
        direction = 'left'
    elif (cat == 'blenderball'):
        direction = 'right'
    elif (cat == 'dirtyball'):
        direction = 'down'

    base_url = 'https://s3.amazonaws.com/mutatorsr/resources/response_images/'
    return base_url + direction + '.png'

def get_meta():
    # Return tabarray with keys 'obj' , 'bg_id', and 'url'
    #assert len(np.unique(selected_basic_objs)) == 4
    # todo: do a better job 
    meta = pk.load(open('meta_pilot11.pkl'))
    return meta


def make_learningperiod_html_data():
    #Get information to make tutorial sequence to be presented at beginning of each HIT:
    tut_combs = [('Beetle', 'ELEPHANT_M', 'z3', 'CGTG_L')] #todo: get rid of the wonky list/tuple indexing
    meta_query = lambda x: x['var'] == 'V3'
    obj_sequencing = [0, 0, 0, 1, 1, 3, 2, 3, 2, 0] # Presentation order of objects

    assert len(obj_sequencing) == tutorial_trials_per_hit
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
                        'urls': [get_url_labeled_resp_img(o1, 1), get_url_labeled_resp_img(o2, 2), get_url_labeled_resp_img(o3, 3), get_url_labeled_resp_img(o4, 4)],
                        'meta': [{'obj': obj} for obj in [o1, o2, o3, o4]],
                        'labels': [o1, o2, o3, o4]
                         }
                        for o1, o2, o3, o4 in tut_combs]

    test_urls = response_info[0]['urls']
    test_meta_entries = response_info[0]['meta']
    response_img_labels = response_info[0]['labels']


    assert len(sample_urls) == len(sample_meta) == tutorial_trials_per_hit
    assert len(test_urls) == len(test_meta_entries) == len(response_img_labels) == len(tut_combs[0])

    learningperiod_html_data = {
        'sample_urls': sample_urls, # List of strings, in order of desired image presentation
        'sample_meta': sample_meta, # List of dicts, one for each image (tabarray dict-ified)
        'tutorial_trials_per_hit': tutorial_trials_per_hit, # integer, set at top of this driver.
        'test_urls': test_urls, # List of strings, urls, one for each option.
        'test_meta_entries': test_meta_entries, # List of dicts, each tabarray dict-ified
        'response_img_labels': response_img_labels, # List of strings ['objname0', 'objname1'...]
    }

    return learningperiod_html_data


def get_exp(sandbox=True, debug=True, dummy_upload=True):
    meta = get_meta()
    urls = meta['url']

    objs = np.unique(meta['obj'])
    objs = objs[:8]
    print '\nObjects to be tested:', objs, '\n'

    #combs = [e for e in itertools.combinations(objs, 4)] # this establishes the direction mappings according to each entry's order. first = up, second = left, etc...
    combs = [tuple(objs[0:4]), tuple(objs[4:8])]

    response_images = []
    response_images.extend([{
                            'urls': [get_url_labeled_resp_img(o1, 1), get_url_labeled_resp_img(o2, 2), get_url_labeled_resp_img(o3, 3), get_url_labeled_resp_img(o4, 4)],
                            'meta': [{'obj': obj} for obj in [o1, o2, o3, o4]],
                            'labels': [o1, o2, o3, o4]
                             }
                            for o1, o2, o3, o4 in combs])

    with open(path.join(path.dirname(__file__), 'tutorial_html_mutatorSR'), 'r') as tutorial_html_file:
        tutorial_html = tutorial_html_file.read()


    if(USE_CANONICAL_IMAGE_RESPONSES == True):
        immutable_response_positions = False
    else:
        immutable_response_positions = True

    html_data = {
        'combs': combs,
        'response_images': response_images,
        'num_trials': num_HITs_per_confusion * TRIALS_PER_HIT, # PER CONFUSION (i.e., each element in combs) to be measured.
        'meta_field': 'obj',
        'meta': meta,
        'urls': urls,
        'shuffle_response_map': True, # Shuffle effector mappings of objects across HITs or no?
        'immutable_response_positions': immutable_response_positions, # Set position of response images, or change according to shuffled effector map ? (turn off for arrows)
        'meta_query' : lambda x: x['var'] == 'V6', # Take images corresponding to all entries meta, rather than meta_query(meta) entries of meta.
        'label_func': lambda x: meta[x]['obj']
    }

    if(tutorial_trials_per_hit>0):
        learningperiod_html_data  = make_learningperiod_html_data()
    else:
        learningperiod_html_data = None

    additionalrules = [{'old': 'LEARNINGPERIODNUMBER',
                        'new':  str(tutorial_trials_per_hit)},
                       {'old': 'OBJTYPE',
                        'new': 'Object Recognition'},
                       {'old': 'TUTORIAL_HTML',
                        'new': tutorial_html},
                       {'old': 'METAFIELD',
                        'new': "'obj'"}]

    exp = StimulusResponseExperiment( # Doesn't need dldata stimulus set; can pass in images / meta explicitly.
            htmlsrc='web/general_mutatorSR.html',
            htmldst='mutatorSR_n%05d.html',
            sandbox=sandbox,
            title='Visual object learning -- learn to categorize strange objects.',
            reward=0.25,
            duration=1600,
            keywords=['neuroscience', 'psychology', 'experiment', 'object recognition'],  # noqa
            description="Complete a visual object recognition task where you learn to associate novel objects with arrow keys.  We expect this HIT to take about 10 minutes, though you must finish in under 25 minutes. By completing this HIT, you understand that you are participating in an experiment for the Massachusetts Institute of Technology (MIT) Department of Brain and Cognitive Sciences. You may quit at any time, and you will remain anonymous. Contact the requester with questions or concerns about this experiment.",  # noqa
            comment='mutatorsr',
            collection_name = 'mutatorsr', #'hvm_basic_2ways',# 'hvm_basic_2ways', # name of MongoDB db to update / create on dicarlo5. Safe to change?
            max_assignments=UNIQUE_WORKERS_PER_HIT, # How many unique workers can complete a particular hit.
            bucket_name='mutatorsr', # Refers to amazon; either mturk or s3 service (I'm not sure.) on which to store htmls. not safe to change before uploading source to bucket
            trials_per_hit= TRIALS_PER_HIT, # sets the periodicity with which to chop up the giant exp._trials into separate .htmls. actual trials; if there is a tutorial that is handled automatically.
            html_data=html_data,
            tmpdir='tmp',
            frame_height_pix=1200,
            othersrc=['../../lib/dltk.js', '../../lib/dltkexpr.js', '../../lib/dltkrsvp.js'],
            additionalrules=additionalrules,
            learningperiod_html_data = learningperiod_html_data, # Information used to insert the initial tutorial at the beginning of each HIT.
            )

    # -- create actual trials
    exp.createTrials(sampling='with-replacement', verbose=1)


    ### Print HIT info
    print '\n'
    img_url_seq = [i[0] for i in exp._trials['imgFiles']]
    print 'Unique images to be tested, including tutorial images:', len(set(img_url_seq))

    # number of times each unique image is assayed:
    rep_counts = [img_url_seq.count(i) for i in set(img_url_seq)]
    print 'Rep counts for each unique image:', rep_counts

    n_total_trials = len(exp._trials['imgFiles'])
    print 'Total trials to be collected:', n_total_trials
    print '\n'


    print '\nRemaining balance on Amazon account:', exp.getBalance()
    print 'Expected cost: $', num_HITs_per_confusion * UNIQUE_WORKERS_PER_HIT * exp.reward * len(combs), '\n'


    ### Return
    if debug:
        return exp, html_data

    print '** Finished creating trials.'


    return exp, html_data


def main(argv=[], partial=False, debug=False):
    sandbox = True


    if len(argv) > 1 and argv[1] == 'download':
        # commandline syntax:
        # ssh -f -N -L 22334:localhost:22334 mil@dicarlo5.mit.edu
        # python driver.py download hitidfname.pkl

        exp = get_exp(sandbox=False, debug=debug)[0] # sandbox must be false
        hitids = pk.load(open(argv[2]))
        print '\n*** Downloading ', len(hitids), 'worker results from mturk and storing in dicarlo5 MongoDB'
        exp.updateDBwithHITs(hitids)
        pk.dump(exp.all_data, open(('results_'+argv[2]), "wb"))
        return exp

        # todo: get HITs that have been completed, approve them based on performance
        # use: get_reviewable_hits(hit_type=None, status='Reviewable', sort_by='Expiration', sort_direction='Ascending', page_size=10, page_number=1)
        # and also, in base.py, def pay_bonuses

    if len(argv) > 1 and argv[1] == 'production':
        sandbox = False
        print '** Creating production HITs'

    else:
        print '** Sandbox mode'
        print '** Enter "[drivername].py production" to publish production HITs.'

    exp = get_exp(sandbox=sandbox, debug=debug)[0]
    exp.prepHTMLs()
    print '** Done prepping htmls.'
    if partial:
        return exp

    exp.testHTMLs()
    print '** Done testing htmls.'
    exp.uploadHTMLs()
    exp.createHIT(secure=True)
    return exp


if __name__ == '__main__':
    main(sys.argv)
