#!/usr/bin/env python
import numpy as np
import cPickle as pk
import tabular as tb
import itertools
import copy
import sys
from mturkutils.base import MatchToSampleFromDLDataExperiment
from os import path
import json

SELECTED_BASIC_OBJS = ['Beetle', 'ELEPHANT_M', 'z3','CGTG_L']
REPEATS_PER_QE_IMG = 0 # Number of times to repeat selected images, specified below, for "Quality Estimation" (i1 internal consistency) purposes
ACTUAL_TRIALS_PER_HIT = 200 # Number of experimental trials, without repeat images. And without tutorial trials.

tutorial_trials_per_hit = 10 # Number of learning trials
num_HITs = 6 # number of HTML files to be generated

def get_url_labeled_resp_img(cat):
    # Get response image for this 'category' cat.
    # Define directions (hardcode for now).
    # TODO: randomize objects and directions, over HITs. 1 HIT/worker
    if(cat == 'Beetle'):
        direction = 'up'
    elif(cat == 'ELEPHANT_M'):
        direction = 'left'
    elif (cat == 'z3'):
        direction = 'right'
    elif (cat == 'CGTG_L'):
        direction = 'down'

    base_url = 'https://s3.amazonaws.com/mutatorsr/resources/response_images/'
    return base_url + direction + '.png'

def get_meta(selected_basic_objs=SELECTED_BASIC_OBJS):
    # Return tabarray with keys 'obj' , 'bg_id', and 'url'
    assert len(np.unique(selected_basic_objs)) == 4
    meta = pk.load(open('meta_pilot4.pkl'))
    return meta


def get_exp(sandbox=True, debug=True, dummy_upload=True):

    #meta = get_meta()
    #urls = meta['url']

    import dldata.stimulus_sets.hvm as hvm
    dataset = hvm.HvMWithDiscfade()
    meta = dataset.meta
    combs = [('Beetle', 'ELEPHANT_M', 'z3','CGTG_L')]
    response_images = []
    response_images.extend([{
                                'urls': [get_url_labeled_resp_img(c1), get_url_labeled_resp_img(c2), get_url_labeled_resp_img(c3), get_url_labeled_resp_img(c4)],
                                'meta': [{'obj': obj} for obj in [c1, c2, c3, c4]],
                                'labels': [c1, c2, c3, c4]
                            } for c1, c2, c3, c4 in combs])

    urls = dataset.publish_images(range(len(dataset.meta)), None, 'hvm_timing',
                                  dummy_upload=dummy_upload)


    with open(path.join(path.dirname(__file__), 'tutorial_html_hvmSR'), 'r') as tutorial_html_file:
        tutorial_html = tutorial_html_file.read()
    label_func = lambda x: hvm.OBJECT_NAMES[x['obj']]
    html_data = {
        'combs': combs,
        'response_images': response_images,
        'num_trials': num_HITs*ACTUAL_TRIALS_PER_HIT,
        'meta_field': 'obj',
        'meta': meta,
        'urls': urls,
        'shuffle_test': False, # Shuffle position of response images or no?
        'meta_query' : lambda x: x['var'] == 'V0', # Take images corresponding to all entries meta, rather than meta_query(meta) entries of meta.
        'label_func': label_func
    }
  #  cat_dict = {'Animals': 'Animal', 'Boats': 'Boat', 'Cars': 'Car',
  #             'Chairs': 'Chair'}#, 'Faces': 'Face', 'Fruits': 'Fruit',
  #             #'Planes': 'Plane', 'Tables': 'Table'}

    additionalrules = [{'old': 'LEARNINGPERIODNUMBER',
                        'new':  str(10)},
                       {'old': 'OBJTYPE',
                        'new': 'Object Recognition'},
                       {'old': 'TUTORIAL_HTML',
                        'new': tutorial_html},
 #                      {'old': 'CATDICT',
 #                       'new': json.dumps(cat_dict)},
                       {'old': 'METAFIELD',
                        'new': "'obj'"}]

    exp = MatchToSampleFromDLDataExperiment( # Doesn't need dldata stimulus set; can pass in images / meta explicitly.
            htmlsrc='web/general_hvmSR.html',
            htmldst='hvmSR_n%05d.html',
            sandbox=sandbox,
            title='Visual object learning --- learn to categorize objects.',
            reward=0.25,
            duration=1600,
            keywords=['neuroscience', 'psychology', 'experiment', 'object recognition'],  # noqa
            description="*Complete a visual object recognition task where you learn to associate objects with arrow keys. We expect this HIT to take about 10 minutes, though you must finish in under 25 minutes. You may complete one HIT in this group. By completing this HIT, you understand that you are participating in an experiment for the Massachusetts Institute of Technology (MIT) Department of Brain and Cognitive Sciences. You may quit at any time, and you will remain anonymous. Contact the requester with questions or concerns about this experiment.",  # noqa
            comment='hvmSR',
            collection_name= None,# 'hvm_basic_2ways', # name of MongoDB db to update / create on dicarlo5. Safe to change?
            max_assignments=1, # How many unique workers can complete a particular hit.
            bucket_name='hvmsr', # Refers to amazon s3 service (I'm not sure.) on which to refer to js sources. not safe to change before uploading source to bucket
            trials_per_hit= ACTUAL_TRIALS_PER_HIT,  # ACTUAL_TRIALS_PER_HIT + (6xREPEATS_PER_QE_IMG) repeats
            html_data=html_data,
            tmpdir='tmp_hvm',
            frame_height_pix=1200,
            othersrc=['../../lib/dltk.js', '../../lib/dltkexpr.js', '../../lib/dltkrsvp.js'],
            additionalrules=additionalrules
            )

    # -- create actual trials
    exp.createTrials(sampling='with-replacement', verbose=1)


    # # TODO Tutorial: Manually prepend tutorial trials with normal objects
    # tutorial_imgData = [{'Sample': meta[0],
    #                      'Test': [{'obj':obj} for obj in list(obj_combs[0])]
    #                      }]#  imgData.append({"Sample": sample_meta_entry, "Test": [test_meta_entry[e] for e in test_objects]})
    # tutorial_imgFiles = [[meta['url'][0],
    #                       [get_url_labeled_resp_img(obj) for obj in list(obj_combs[0])]]
    #                      ] #.append([sample_url, [test_url[e] for e in test_objects]]) for all trials
    # tutorial_labels = list(obj_combs[0]) # .append(labelfunc(test_meta_entry[e]) for e in test_objects))
    #
    #
    # exp._trials['imgData'] = tutorial_imgData + exp._trials['imgData']
    # exp._trials['imgFiles'] = tutorial_imgFiles + exp._trials['imgFiles']  # tutorial_imgFiles: list
    # exp._trials['labels'] = tutorial_labels + exp._trials['labels']


    ### Print HIT info ###
    print '\n'
    img_url_seq = [i[0] for i in exp._trials['imgFiles']]
    print 'Unique images to be tested:', len(set(img_url_seq))

    # number of times each unique image is assayed:
    rep_counts = [img_url_seq.count(i) for i in set(img_url_seq)]
    print 'Rep counts for each unique image:', rep_counts

    n_total_trials = len(exp._trials['imgFiles'])
    print 'Total trials to be collected:', n_total_trials
    print '\n'




    if debug:
        return exp, html_data


    ## TODO: use this code to make the tutorial with other kinds of objects.

    # # -- in each HIT, the followings will be repeated 4 times to
    # # estimate "quality" of data
    # ind_repeats = [] #[0, 4, 47, 9, 17, 18] * REPEATS_PER_QE_IMG
    # rng = np.random.RandomState(0)
    # rng.shuffle(ind_repeats)
    # trials_qe = {e: [copy.deepcopy(exp._trials[e][r]) for r in ind_repeats]
    #              for e in exp._trials}
    #
    # # -- flip answer choices of some repeated images
    # n_qe = len(trials_qe['labels'])
    # # if True, flip
    # flips = [True] * (n_qe / 2) + [False] * (n_qe - n_qe / 2)
    # assert len(flips) == n_qe
    # rng.shuffle(flips)
    # assert len(trials_qe.keys()) == 4
    #
    # for i, flip in enumerate(flips):
    #     if not flip:
    #         continue
    #     trials_qe['imgFiles'][i][1].reverse()
    #     trials_qe['labels'][i].reverse()
    #     trials_qe['imgData'][i]['Test'].reverse()
    #
    # # -- actual application
    # offsets = np.arange(
    #     ACTUAL_TRIALS_PER_HIT - 3, -1,
    #     -ACTUAL_TRIALS_PER_HIT / float(len(ind_repeats))
    # ).round().astype('int')
    # assert len(offsets) == len(offsets)
    #
    # n_hits_floor = n_total_trials / ACTUAL_TRIALS_PER_HIT
    # n_applied_hits = 0
    # for i_trial_begin in xrange((n_hits_floor - 1) * ACTUAL_TRIALS_PER_HIT,
    #                             -1, -ACTUAL_TRIALS_PER_HIT):
    #     for k in trials_qe:
    #         for i, offset in enumerate(offsets):
    #             exp._trials[k].insert(i_trial_begin + offset, trials_qe[k][i])
    #             # exp._trials[k].insert(i_trial_begin + offset, 'test')
    #     n_applied_hits += 1
    #
    # print '** n_applied_hits =', n_applied_hits
    # print '** len for each in _trials =', \
    #     {e: len(exp._trials[e]) for e in exp._trials}
    #
    # # -- sanity check
    # assert 50 == n_applied_hits, n_applied_hits
    # assert len(exp._trials['imgFiles']) == 50 * 164
    # s_ref_labels = [tuple(e) for e in trials_qe['labels']]
    # offsets2 = np.arange(24)[::-1] + offsets
    # ibie = zip(range(0, 50 * 164, 164), range(164, 50 * 164, 164))
    # assert all(
    #     [[(e1, e2) for e1, e2 in
    #      np.array(exp._trials['labels'][ib:ie])[offsets2]] == s_ref_labels
    #      for ib, ie in ibie])
    #
    # # -- drop unneeded, potentially abusable stuffs
    # #del exp._trials['imgData']
    print '** Finished creating trials.'

    return exp, html_data


def main(argv=[], partial=False, debug=False):
    sandbox = True

    if len(argv) > 1 and argv[1] == 'download':
    ## Todo: figure this out
        # commandline syntax:
        # ssh -f -N -L 22334:localhost:22334 <username>@dicarlo5.mit.edu
        # python driver.py download hitidfname.pkl
        exp = get_exp(sandbox=sandbox, debug=debug)[0]
        #hitids = pk.load(open(argv[2]))
        #print len(hitids)
        print 'hey'
        print '\n*** Downloading worker results from mturk and storing in dicarlo5 MongoDB'
        exp.updateDBwithHITs(['32XN26MTXZTRR3AG73Q3WWBC6WJL0P'])
        pk.dump(exp.all_data, open('results.pkl', "wb"))
        return exp


    if len(argv) > 1 and argv[1] == 'production':
        sandbox = False
        print '** Creating production HITs'
    else:
        print '** Sandbox mode'
        print '** Enter "driver.py production" to publish production HITs.'

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
