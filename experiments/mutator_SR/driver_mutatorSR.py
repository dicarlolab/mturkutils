#!/usr/bin/env python
local_library_path = '/mindhive/dicarlolab/u/mil/'
import sys
import dldata.stimulus_sets.hvm as hvm #Don't remove, or else for spooky reasons "import pymongo" doesn't work
import numpy as np
import cPickle as pk
import tabular as tb
import itertools
import copy
import sys
from mturkutils.base import StimulusResponseExperiment
from os import path
import json
import random
import math
import make_SRtutorialhtml

# Todo: add a canvas for these. The solution right now is hacky - the response_image_indices for these stay the same (1, 2, 3, 4) but the labels are shuffled for every HIT.
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

def get_url_canonical_resp_img(obj):
    base_url = 'https://s3.amazonaws.com/mutatorsr/resources/response_images/'
    return base_url + 'pilot11_canonical/_gray_'+obj + '.png'

def get_url_abstract_resp_img(obj_num):
    base_url = 'https://s3.amazonaws.com/mutatorsr/resources/response_images/'
    return base_url + 'abstract' + str(obj_num) + '.png'

def get_exp(exp_spec, sandbox=True, debug=True, dummy_upload=True):

    nickname = exp_spec['nickname_prefix']
    meta = exp_spec['meta']
    urls = exp_spec['image_urls']

    _mquery_field = exp_spec['meta_query_tuple'][0]
    _mquery_fieldmatch = exp_spec['meta_query_tuple'][1]
    meta_query = lambda x: x[_mquery_field] == _mquery_fieldmatch

    labelfunc = lambda x: x[exp_spec['labelfunc_field']]

    labelset = exp_spec['labelset']
    combs = exp_spec['confusions'] # [tuple(objs[0:4]), tuple(objs[4:8])]

    trials_per_hit = exp_spec['trials_per_hit']
    trials_per_confusion = exp_spec['withinHIT_reps_per_image']*exp_spec['trials_per_hit']#*exp_spec['overPop_min_reps_per_image']
    unique_workers_per_hit = exp_spec['overPop_min_reps_per_image']

    immutable_response_positions = not exp_spec['shuffle_response_positions']

    num_HITs_per_confusion = np.true_divide(trials_per_confusion, trials_per_hit)

    response_images = []
    response_images.extend([{
                            'urls': [get_url_arrow_response_img(1), get_url_arrow_response_img(2), get_url_arrow_response_img(3), get_url_arrow_response_img(4)],
                            'meta': [{'obj': obj} for obj in [o1, o2, o3, o4]],
                            'labels': [o1, o2, o3, o4]
                             }
                            for o1, o2, o3, o4 in combs])     # todo: remove implicit dependency on numobjects = 4 = len(combs[i])

    with open(path.join(path.dirname(__file__), 'tutorial_html_hvmSR'), 'r') as tutorial_html_file:
        tutorial_html = tutorial_html_file.read()

    html_data = {
        'combs': combs,
        'response_images': response_images,
        'num_trials': trials_per_confusion, # TRIALS PER CONFUSION (i.e., each element in combs) to be measured.
        'meta_field': 'obj', # todo: implicit dependency on labelfunc = lambda x: x['obj']
        'meta': meta,
        'urls': urls,
        'shuffle_response_map': True, # Shuffle effector mappings of objects across HITs or no?
        'immutable_response_positions': immutable_response_positions, # Set position of response images, or change according to shuffled effector map ? (turn off for arrows)
        'meta_query' : meta_query, # Take images corresponding to all entries meta, rather than meta_query(meta) entries of meta.
        'label_func': labelfunc,
    }

    num_tutorial_trials = exp_spec['number_tutorial_trials']
    if(num_tutorial_trials):
        tutorial_type = exp_spec['tutorial_content']
        learningperiod_html_data  = make_SRtutorialhtml.make(tutorial_type, num_tutorial_trials)
    else:
        learningperiod_html_data = None

    additionalrules = [{'old': 'LEARNINGPERIODNUMBER',
                        'new':  str(num_tutorial_trials)},
                       {'old': 'OBJTYPE',
                        'new': 'Object Recognition'},
                       {'old': 'TUTORIAL_HTML',
                        'new': tutorial_html},
                       {'old': 'METAFIELD',
                        'new': "'obj'"}]

    exp = StimulusResponseExperiment( # Doesn't need dldata stimulus set; can pass in images / meta explicitly.
            htmlsrc='web/general_mutatorSR.html',
            htmldst=nickname+'_n%05d.html',
            sandbox=sandbox,
            title=exp_spec['Amazon_title'],
            reward=0.15,
            duration=1600,
            keywords=['neuroscience', 'psychology', 'experiment', 'object recognition'],  # noqa
            description=exp_spec['Amazon_description'],  # noqa
            comment=nickname,
            collection_name = 'mutatorsr', #'hvm_basic_2ways',# 'hvm_basic_2ways', # name of MongoDB db to update / create on dicarlo5. Safe to change?
            max_assignments=unique_workers_per_hit, # How many unique workers can complete a particular hit.
            bucket_name='mutatorsr', # Refers to amazon; either mturk or s3 service (I'm not sure.) on which to store htmls. not safe to change before uploading source to bucket
            trials_per_hit= trials_per_hit, # sets the periodicity with which to chop up the giant exp._trials into separate .htmls. actual trials; if there is a tutorial that is handled automatically.
            html_data=html_data,
            tmpdir='tmp',
            frame_height_pix=1200,
            othersrc=['../../lib/dltk.js', '../../lib/dltkexpr.js', '../../lib/dltkrsvp.js'],
            additionalrules=additionalrules,
            learningperiod_html_data = learningperiod_html_data, # Information used to insert the initial tutorial at the beginning of each HIT.
            )

    # -- create actual trials
   
    exp.createTrials(sampling=exp_spec['sampling_method'], verbose=2)
    #exp._trials = unpack_into_trials(pk.load(open('params_mut_2wayV62016-11-13--23.48.34.pk', 'rb')))
    print '\n\n'

    ### Print HIT info
    print '\n'
    #print '\nLabels to be tested:', labelset
    img_url_seq = [i[0] for i in exp._trials['imgFiles']]
    print 'Number of unique images to be tested, including tutorial images:', len(set(img_url_seq))

    # number of times each unique image is assayed:
    rep_counts = [img_url_seq.count(i) for i in set(img_url_seq)]
    print len(rep_counts)
    print 'Rep counts for each unique image:', rep_counts

    n_total_trials = len(exp._trials['imgFiles'])
    print 'Total number of HIT trials:', n_total_trials
    print 'Total number of HITs: ', np.divide(n_total_trials, trials_per_hit+num_tutorial_trials)
    print 'Total number of workers per HIT:', unique_workers_per_hit
    print '\n'


    print '\nRemaining balance on Amazon account:', exp.getBalance()
    print 'Expected cost: $', num_HITs_per_confusion * unique_workers_per_hit * exp.reward * len(combs), '\n'

    ### Return
    print '** Finished creating trials.'

    return exp, html_data

def unpack_into_trials(HIT_dictionary): 
    # Unpack experiment parameters and write _trials in mturkutils.base.exp form
    print 'Number of HITs to write out:', len(HIT_dictionary.keys())

    # Output: 
    imgs = [] # List of [sample_url, [test_urls[e] for e in response_image_indices]]. # Position of urls within second list determines position on screen.
    imgData = [] # List of: {
                  #          "Sample": sample_meta,
                  #          "Test": [test_meta_entries[e] for e in response_image_indices]}
    labels = [] # [response_img_labels[e] for e in response_image_indices] 
    meta_fields = [] # [str,...]

    for HIT in HIT_dictionary.keys(): 
        h = HIT_dictionary[HIT]
        imgs.extend([[i, j] for i, j in zip(h['sample_urls'], h['test_urls'])])


        meta_dictify = lambda meta_entry: {name: value for name, value in
                     zip(meta_entry.dtype.names, meta_entry.tolist())}


        #assert True == False
        imgData.extend([{'Sample': meta_dictify(sample_meta), 'Test':test_metas} for sample_meta, test_metas in zip(h['sample_metas'], h['test_metas'])])
        



        labels.extend(h['test_labels'])
        meta_fields.extend(h['meta_field'])

    _trials = {'imgFiles': imgs, 'imgData': imgData, 'labels': labels,
                               'meta_field': meta_fields}
    return _trials 

def main(argv=[], partial=True, debug=False):

    # Command line usage: 
    ## python driver_mutatorSR.py [paramsfile.pk]
    ## python driver_mutatorSR.py [paramsfile.pk] production

    ## ssh -f -N -L 22334:localhost:22334 mil@dicarlo5.mit.edu
    # python driver_mutatorSR.py [paramsfile.pk] download [hitidname.pkl]

    if(len(argv)<1):
        print 'Need to specify params file.'
        return

    EXPERIMENT_PARAM_FILE = argv[1] 
    print '\n Using params at: '+EXPERIMENT_PARAM_FILE
    exp_spec = pk.load(open(EXPERIMENT_PARAM_FILE))

    if len(argv) > 2 and argv[2] == 'download':
        exp = get_exp(exp_spec, sandbox=False, debug=debug)[0] 
        hitids = pk.load(open(argv[3]))
        print '\n*** Downloading ', len(hitids), 'HITID results from mturk and storing in dicarlo5 MongoDB'
        exp.updateDBwithHITs(hitids)
        pk.dump(exp.all_data, open(('./result_pickles/results_'+exp_spec['nickname_prefix']+argv[3]), "wb"))
        return exp

    if len(argv) > 2 and argv[2] == 'production':
        sandbox = False
        print '** Creating production HITs **'

    else:
        sandbox = True
        print '** Sandbox mode **'

    exp = get_exp(exp_spec, sandbox=sandbox, debug=debug)[0]
    exp.prepHTMLs()

    print '** Done prepping HTMLs. **'
    if partial:
        return exp

    exp.testHTMLs()
    print '** Done testing HTMLs. **'
    exp.uploadHTMLs()
    exp.createHIT(secure=True)
    return exp

if __name__ == '__main__':
    main(sys.argv)
