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
import os
import json
import random
import math
from time import gmtime, strftime



def get_exp(experiment_params, sandbox=True, debug=True, dummy_upload=True):
    experiment_meta = experiment_params[0]
    HIT_dictionary = experiment_params[1]
    
    assert len(experiment_params) == 2
    assert len(set([len(HIT_dictionary[k]['sample_urls']) for k in HIT_dictionary.keys()])) == 1 # all the same # of trials per HIT

    # Settings to load from interface:
    unique_workers_per_hit = experiment_meta['workers_per_HIT']
    nickname = experiment_meta['nickname']
    htmlsrc = experiment_meta['html_src']
    instructions_htmlsrc = experiment_meta['instructions_html_src']
    num_tutorial_trials = experiment_meta['num_tutorial_trials']
    hit_id_save_directory = experiment_meta['hit_id_save_directory']
    if not os.path.exists(hit_id_save_directory): os.makedirs(hit_id_save_directory)
    trials_per_hit = len(HIT_dictionary[HIT_dictionary.keys()[0]]['sample_urls'])
    reward_per_hit = experiment_meta['reward_per_HIT']
    target_bonus_per_hit = experiment_meta['target_bonus_per_HIT']
    

    with open(instructions_htmlsrc, 'r') as t:
        tutorial_html = t.read()

    blurb = "Complete a visual object recognition task where you learn to associate 3D objects with arrow keys.\n" \
            "  We expect this HIT to take about 10 minutes, though you must finish in under 25 minutes. \n" \
            "By completing this HIT, you understand that you are participating in an experiment\n " \
            "for the Massachusetts Institute of Technology (MIT) Department of Brain and Cognitive Sciences. \n" \
            "You may quit at any time, and you will remain anonymous. \n" \
            "Contact the requester with questions or concerns about this experiment.\n",

    additionalrules = [{'old': 'LEARNINGPERIODNUMBER',
                        'new':  str(num_tutorial_trials)},
                       {'old': 'OBJTYPE',
                        'new': 'Object Recognition'},
                       {'old': 'TUTORIAL_HTML',
                        'new': tutorial_html},
                       {'old': 'METAFIELD',
                        'new': "'obj'"}, 
                        {'old': 'TARGETBONUSNUMBER', 
                        'new': str(target_bonus_per_hit)}]


    html_tmp_dir = os.path.join('tmp_', nickname+'_'+strftime("%m%d%Y%H%M%S", gmtime()))
    if not os.path.exists(html_tmp_dir): 
        print('Making tmp dir at', html_tmp_dir)
        os.makedirs(html_tmp_dir)
    print('Saving temporary htmls at', html_tmp_dir)


    exp = StimulusResponseExperiment( 
            htmlsrc=htmlsrc,
            htmldst=nickname+'_'+strftime("%m%d%Y%H%M%S", gmtime())+'_SR_n%05d.html',
            sandbox=sandbox,
            title='Visual object learning - learn to categorize 3D objects. (' + strftime("%m/%d/%Y--%H:%M:%S", gmtime())+')',
            reward=reward_per_hit,
            duration=1600,
            keywords=['neuroscience', 'psychology', 'experiment', 'object recognition'],  # noqa
            description=blurb,  # noqa
            comment=nickname,
            log_prefix = hit_id_save_directory, 
            collection_name = 'mutatorsr', 
            max_assignments=unique_workers_per_hit, # How many unique workers can complete a particular hit.
            bucket_name='mutatorsr', # Refers to amazon; either mturk or s3 service (I'm not sure.) on which to store htmls. not safe to change before uploading source to bucket
            trials_per_hit= trials_per_hit, 
            html_data=None,
            tmpdir=html_tmp_dir,
            frame_height_pix=1200,
            othersrc=['../../lib/dltk.js', '../../lib/dltkexpr.js', '../../lib/dltkrsvp.js'],
            additionalrules=additionalrules)

    # Load trials from disk 
    exp._trials = unpack_into_trials(HIT_dictionary)


    # Print diagnostic info
    print '\n'
    img_url_seq = [i[0] for i in exp._trials['imgFiles']]

    n_total_trials = len(exp._trials['imgFiles'])
    n_total_experimental_trials = np.divide(n_total_trials, trials_per_hit) * (trials_per_hit - num_tutorial_trials)
    n_total_learning_trials = np.divide(n_total_trials, trials_per_hit) * (num_tutorial_trials)
    n_hits = np.divide(n_total_trials, trials_per_hit)
    assert n_total_trials == n_total_learning_trials + n_total_experimental_trials

    rep_counts = [img_url_seq.count(i) for i in set(img_url_seq)]
    print '# HITs: ', n_hits
    print 'Workers per HIT:', unique_workers_per_hit
    print 'Population reps per unique image:', np.array(sorted(rep_counts))*unique_workers_per_hit
    print '# experimental trials:', n_total_experimental_trials
    print '# learning trials:', n_total_learning_trials
    print 'Total trials:', n_total_trials
    print '\nRemaining balance on Amazon account:', exp.getBalance()
    print 'Expected cost: $', n_hits* exp.reward, '\n'

    return exp


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


def main(argv=[], partial=False, debug=False):

    # Command line usage: 
    ## python driver_mutatorSR.py [paramsfile.pk]
    ## python driver_mutatorSR.py [paramsfile.pk] production

    ## ssh -f -N -L 22334:localhost:22334 mil@dicarlo5.mit.edu
    # python driver_mutatorSR.py [paramsfile.pk] download [hitidname.pkl]

    if(len(argv)<1):
        print 'Need to specify params file.'
        return

    EXPERIMENT_PARAM_FILE_PATH = argv[1] 
    print '\n Using params at: '+EXPERIMENT_PARAM_FILE_PATH
    experiment_params = pk.load(open(EXPERIMENT_PARAM_FILE_PATH, 'rb')) #pk.load(open(EXPERIMENT_PARAM_FILE))


    if len(argv) > 2 and argv[2] == 'download':
        exp = get_exp(experiment_params, sandbox=False, debug=debug)[0] 
        hitids = pk.load(open(argv[3]))
        print '\n*** Downloading ', len(hitids), 'HITID results from mturk and storing in dicarlo5 MongoDB'
        exp.updateDBwithHITs(hitids)
        pk.dump(exp.all_data, open(('./result_pickles/results_'+experiment_params[0]['nickname']+argv[3]), "wb"))
        return exp

    
    if len(argv) > 2 and argv[2] == 'production':
        sandbox = False
        print '** Creating production HITs **'

    else:
        sandbox = True
        print '** Sandbox mode **'

    exp = get_exp(experiment_params, sandbox=sandbox, debug=debug)
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
