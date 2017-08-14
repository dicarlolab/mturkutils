import numpy as np
import sys 

local_library_path = '/mindhive/dicarlolab/u/mil/'
sys.path.insert(1,local_library_path)

import mldata.mldata.sets.mut as mut
import dldata.stimulus_sets.hvm as hvm
import cPickle as pk
import os
import collections
from time import gmtime, strftime
import itertools
from tqdm import tqdm


SAVE_DIRECTORY = './outputs'

def write_mturk_SRexperiment(): 
    

    rng_seed = 0
    np.random.seed(rng_seed)    

    # User meta settings: 
    #htmlsrc = 'web/general_SR_2way.html' # template file
    htmlsrc = 'web/general_SR_2way_nb.html' # template file
    instructions_html_src = 'web/instruction_html_SR2way.html' # Instruction click-through screens 
    experiment_nickname = 'mut_Batch0_2way_scramble150_stoch0_obj0_1_cbalance.pk' # Determines outputs/[folder] in which everything will be saved
    experiment_param_save_directory = os.path.join(SAVE_DIRECTORY, experiment_nickname, 'params')
    psychophysics_save_directory = os.path.join(SAVE_DIRECTORY, experiment_nickname, 'psychophysics_data')
    hit_id_save_directory = os.path.join(SAVE_DIRECTORY, experiment_nickname, 'hit_ids/')
    

    print hit_id_save_directory


    # User HIT settings
    #d = hvm.HvMWithDiscfade()
    #d = mut.Pilot18()
    d = mut.Batch0()

    meta  = d.meta
    number_of_workers_per_HIT = 30
    number_of_HITs_per_comb = 1
    reward_per_HIT = 0.20
    target_bonus_per_HIT = 0
    nways = 2
    num_tutorial_trials = 10
    number_positive_exemplars_per_label_in_HIT = 100
    meta_query = lambda x: x['var'] == 'V3' #or x['var'] == 'V6'
    meta_field = 'obj' 

    EFFECTOR_SCRAMBLE = True
    OBJECT_SCRAMBLE = False 
    COUNTERBALANCE_CONDITION = False # switch order of object pairs
    EFFECTOR_SWITCH_TRIAL = number_positive_exemplars_per_label_in_HIT

    if OBJECT_SCRAMBLE == True: 
        EFFECTOR_SWITCH_TRIAL = number_positive_exemplars_per_label_in_HIT * 2 
    
    #testable_objects = set(meta[meta_field])
    
    #hvm_testable_objects = [ 'bear', 'ELEPHANT_M', '_18', 'face0001', 'alfa155', 'breed_pug', 'TURTLE_L', 'Apple_Fruit_obj',  'f16', '_001', '_014', 'face0002']
    
    #mut18_testable_objects = ['ballpile', 'basket', 'blenderball', 'blowfish', 'bouquet', 'dippindots', 'moth', 
    #'octopus', 'pinecone', 'slug', 'spikybeast', 'staircase']

    testable_objects = ['batch0obj0', 'batch0obj1', 'batch0obj21', 'batch0obj26', 'batch0obj80', 'batch0obj82']
    #scramble_objects = ['batch0obj4', 'batch0obj5']


    #combs = [e for e in itertools.combinations(testable_objects, nways)]
    combs = [('batch0obj0', 'batch0obj1'), 
    ('batch0obj21', 'batch0obj26'), 
    ('batch0obj80', 'batch0obj82')]
    #scramble_combs = [('batch0obj4', 'batch0obj5')]




    # remove previously tested combs 
    '''
    previous_combs = [{'_001', 'f16'},
 {'TURTLE_L', 'face0002'},
 {'_18', 'bear'},
 {'ELEPHANT_M', '_001'},
 {'ELEPHANT_M', 'TURTLE_L'},
 {'alfa155', 'f16'},
 {'_014', 'f16'},
 {'TURTLE_L', 'f16'},
 {'breed_pug', 'face0001'},
 {'ELEPHANT_M', 'alfa155'},
 {'Apple_Fruit_obj', 'TURTLE_L'},
 {'_001', 'bear'},
 {'_18', 'breed_pug'},
 {'ELEPHANT_M', 'breed_pug'},
 {'_18', 'face0001'},
 {'_014', 'face0001'},
 {'_18', 'f16'},
 {'Apple_Fruit_obj', '_014'},
 {'_001', 'breed_pug'},
 {'_014', 'bear'},
 {'alfa155', 'bear'},
 {'alfa155', 'face0002'},
 {'breed_pug', 'face0002'},
 {'_014', 'alfa155'},
 {'face0001', 'face0002'},
 {'f16', 'face0002'},
 {'_001', 'face0001'},
 {'Apple_Fruit_obj', '_001'},
 {'ELEPHANT_M', 'bear'},
 {'Apple_Fruit_obj', '_18'}]

    
    previous_combs = [('ballpile', 'spikybeast'), 
    ('basket', 'staircase'), 
    ('dippindots', 'slug'), 
    ('blowfish', 'pinecone'), 
    ('blenderball', 'bouquet'), 
    ('moth', 'octopus')]
    previous_combs = [set(x) for x in previous_combs]
    '''
    previous_combs = []

    remove_counter = 0 
    for c in previous_combs: 
        for potential_comb in combs: 
            if(set(potential_comb) == c): 
                combs.remove(potential_comb)
                print('Removed', potential_comb)
                remove_counter+=1 
        
    assert remove_counter == len(previous_combs)
    
    #all_sample_urls = ['https://s3.amazonaws.com/mutatorsr/resources/mutator18_stimuli/discfaded/'+h for h in meta['id']] # in meta order
    #all_sample_urls = d.publish_images(range(len(d.meta)), None, 'hvm_timing', dummy_upload=True)
    all_sample_urls = ['https://s3.amazonaws.com/mutatorsr/resources/Batch0/'+h for h in meta['id']] # in meta order]

    test_url = ['https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png', 
                'https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png'] # up / left / right / down is the default order
    all_effector_maps = [np.random.permutation(nways) for i in range(len(combs)*number_of_HITs_per_comb)] # One map per HIT;;  indexes into comb[i]
    

    ### Write HITs
    tut_sample_urls, tut_sample_metas, tut_test_urls, tut_test_metas, tut_test_labels, tut_meta_fields  = _make_abstractshape_tutorial_trials(num_trials = num_tutorial_trials, nways = nways, rng_seed = rng_seed); 
    HIT_dictionary = {}
    query_indices = set(filter(lambda x: meta_query(meta[x]), range(len(meta))))
    for comb_counter, comb in enumerate(tqdm(combs)):
        
        for label in comb: 
        # Get all urls / meta for this label, that passes meta_query. 
        # And mix their order. 
            label_indices = set(filter(lambda x: meta[x][meta_field] == label, range(len(meta))))
            eligible_label_indices = list(query_indices & label_indices)
            shuffled_indices = list(np.random.permutation(eligible_label_indices))

            # Divvy up the label's exemplars amongst the HITs. 
            try: 
                assert number_positive_exemplars_per_label_in_HIT * number_of_HITs_per_comb <= len(shuffled_indices)
            except: 
                print 'Current label:', label 
                print 'Desired # of HITs per comb:',  number_of_HITs_per_comb
                print 'Desired # of positive exemplars per label in a HIT:', number_positive_exemplars_per_label_in_HIT
                print 'Necessary # positive exemplars:', number_positive_exemplars_per_label_in_HIT * number_of_HITs_per_comb
                print 'Total available positive exemplars:', len(shuffled_indices)
                raise Exception


            for HIT_number in range(number_of_HITs_per_comb): 
                index_lb = HIT_number*number_positive_exemplars_per_label_in_HIT
                index_ub = HIT_number*number_positive_exemplars_per_label_in_HIT+number_positive_exemplars_per_label_in_HIT
                HIT_indices = shuffled_indices[index_lb:index_ub]

                # Get appropriate exemplars
                HIT_label_sample_urls = [all_sample_urls[i] for i in HIT_indices]
                HIT_label_sample_meta = [meta[i] for i in HIT_indices]

                # Record in dictionary
                if (comb, HIT_number) not in HIT_dictionary: HIT_dictionary[(comb, HIT_number)] = collections.defaultdict(list)
                HIT_dictionary[(comb, HIT_number)]['sample_urls'].extend(HIT_label_sample_urls)
                HIT_dictionary[(comb, HIT_number)]['sample_metas'].extend(HIT_label_sample_meta)
        
        if(OBJECT_SCRAMBLE): 
            print 'Adding trials for object scramble...'
            for label in scramble_combs[comb_counter]: 
                label_indices = set(filter(lambda x: meta[x][meta_field] == label, range(len(meta))))
                eligible_label_indices = list(query_indices & label_indices)
                shuffled_indices = list(np.random.permutation(eligible_label_indices))

                # Divvy up the label's exemplars amongst the HITs. 
                try: 
                    assert number_positive_exemplars_per_label_in_HIT * number_of_HITs_per_comb <= len(shuffled_indices)
                except: 
                    print 'Current label:', label 
                    print 'Desired # of HITs per comb:',  number_of_HITs_per_comb
                    print 'Desired # of positive exemplars per label in a HIT:', number_positive_exemplars_per_label_in_HIT
                    print 'Necessary # positive exemplars:', number_positive_exemplars_per_label_in_HIT * number_of_HITs_per_comb
                    print 'Total available positive exemplars:', len(shuffled_indices)
                    raise Exception


                for HIT_number in range(number_of_HITs_per_comb): 
                    index_lb = HIT_number*number_positive_exemplars_per_label_in_HIT
                    index_ub = HIT_number*number_positive_exemplars_per_label_in_HIT+number_positive_exemplars_per_label_in_HIT
                    HIT_indices = shuffled_indices[index_lb:index_ub]

                    # Get appropriate exemplars
                    HIT_label_sample_urls = [all_sample_urls[i] for i in HIT_indices]
                    HIT_label_sample_meta = [meta[i] for i in HIT_indices]

                    # Record in dictionary
                    if (comb, HIT_number) not in HIT_dictionary: HIT_dictionary[(comb, HIT_number)] = collections.defaultdict(list)
                    HIT_dictionary[(comb, HIT_number)]['sample_urls'].extend(HIT_label_sample_urls)
                    HIT_dictionary[(comb, HIT_number)]['sample_metas'].extend(HIT_label_sample_meta)
        


        # Shuffle HIT trials to mix labels temporally
        for HIT_number in range(number_of_HITs_per_comb): 
            HIT_absolute_counter = comb_counter*number_of_HITs_per_comb+HIT_number

            n_experimental_trials = len(HIT_dictionary[(comb, HIT_number)]['sample_urls'])


            HIT_trial_order = np.random.permutation(n_experimental_trials) # Order of trials

            if(OBJECT_SCRAMBLE): 
                # preserve order of first / second halves (keep shuffling within halves)
                if(n_experimental_trials % 2 != 0): 
                    print 'odd number of trials is not yet supported lol'
                    raise NotImplementedError
                print n_experimental_trials
                HIT_trial_order = np.array(list(np.random.permutation(n_experimental_trials/2)) + list(n_experimental_trials/2+np.random.permutation(n_experimental_trials/2)))
                if(COUNTERBALANCE_CONDITION): 
                    print 'Counterbalance condition!'
                    HIT_trial_order = np.array(list(n_experimental_trials/2+np.random.permutation(n_experimental_trials/2))+list(np.random.permutation(n_experimental_trials/2)) )


                print HIT_trial_order
                
            elif(EFFECTOR_SCRAMBLE and (not OBJECT_SCRAMBLE) and COUNTERBALANCE_CONDITION):
                print 'Counterbalancing effector switch'
                # switch order of object pairs
                if(n_experimental_trials % 2 != 0): 
                    print 'odd number of trials is not yet supported lol'
                    raise NotImplementedError

                HIT_trial_order_2 = np.array(list(HIT_trial_order[len(HIT_trial_order)/2:])+list(HIT_trial_order[0:len(HIT_trial_order)/2]))
                HIT_trial_order = HIT_trial_order_2

                for p in [HIT_dictionary[(comb, HIT_number)]['sample_urls'][h] for h in HIT_trial_order][100:110]: 
                    print p
      

            # Shuffle order (currently o1, o2... contiguous). 
            HIT_dictionary[(comb, HIT_number)]['sample_urls'] = [HIT_dictionary[(comb, HIT_number)]['sample_urls'][h] for h in HIT_trial_order]
            HIT_dictionary[(comb, HIT_number)]['sample_metas'] = [HIT_dictionary[(comb, HIT_number)]['sample_metas'][h] for h in HIT_trial_order]

            # Also add test images/meta.
            effector_map = all_effector_maps[HIT_absolute_counter]
            comb_test_url = test_url # here would be the appropriate place to make alterations for m2s

            # if no effector scrambling
            if (EFFECTOR_SCRAMBLE == False): 
                HIT_dictionary[(comb, HIT_number)]['test_urls'] = [[turl for turl in comb_test_url] for i in range(n_experimental_trials)]
                HIT_dictionary[(comb, HIT_number)]['test_metas'] = [[{meta_field: comb[j]} for j in effector_map] for i in range(n_experimental_trials)] 
                HIT_dictionary[(comb, HIT_number)]['test_labels'] = [[comb[j] for j in effector_map] for i in range(n_experimental_trials)]
            elif (EFFECTOR_SCRAMBLE == True and OBJECT_SCRAMBLE == False): 
                print 'Scrambling map at', EFFECTOR_SWITCH_TRIAL
                # First effector map 
                HIT_dictionary[(comb, HIT_number)]['test_urls'] = [[turl for turl in comb_test_url] for i in range(EFFECTOR_SWITCH_TRIAL)]
                HIT_dictionary[(comb, HIT_number)]['test_metas'] = [[{meta_field: comb[j]} for j in effector_map] for i in range(EFFECTOR_SWITCH_TRIAL)] 
                HIT_dictionary[(comb, HIT_number)]['test_labels'] = [[comb[j] for j in effector_map] for i in range(EFFECTOR_SWITCH_TRIAL)]

                # Second effector map
                HIT_dictionary[(comb, HIT_number)]['test_urls'].extend([[turl for turl in comb_test_url] for i in range(EFFECTOR_SWITCH_TRIAL)])
                HIT_dictionary[(comb, HIT_number)]['test_metas'].extend([[{meta_field: comb[j]} for j in reversed(effector_map)] for i in range(EFFECTOR_SWITCH_TRIAL)])
                HIT_dictionary[(comb, HIT_number)]['test_labels'].extend([[comb[j] for j in reversed(effector_map)] for i in range(EFFECTOR_SWITCH_TRIAL)])
            elif (OBJECT_SCRAMBLE == True): 
                print 'Switching objects at', EFFECTOR_SWITCH_TRIAL

                 

                # First object set 
                HIT_dictionary[(comb, HIT_number)]['test_urls'] = [[turl for turl in comb_test_url] for i in range(EFFECTOR_SWITCH_TRIAL)]
                HIT_dictionary[(comb, HIT_number)]['test_metas'] = [[{meta_field: comb[j]} for j in effector_map] for i in range(EFFECTOR_SWITCH_TRIAL)] 
                HIT_dictionary[(comb, HIT_number)]['test_labels'] = [[comb[j] for j in effector_map] for i in range(EFFECTOR_SWITCH_TRIAL)]

                # Second object set
                HIT_dictionary[(comb, HIT_number)]['test_urls'].extend([[turl for turl in comb_test_url] for i in range(EFFECTOR_SWITCH_TRIAL)])
                HIT_dictionary[(comb, HIT_number)]['test_metas'].extend([[{meta_field: scramble_combs[comb_counter][j]} for j in reversed(effector_map)] for i in range(EFFECTOR_SWITCH_TRIAL)])
                HIT_dictionary[(comb, HIT_number)]['test_labels'].extend([[scramble_combs[comb_counter][j] for j in effector_map] for i in range(EFFECTOR_SWITCH_TRIAL)])

                if(COUNTERBALANCE_CONDITION):
                    print 'Counter balance condition (2/2)'
                    # Second object set
                    HIT_dictionary[(comb, HIT_number)]['test_urls'] = [[turl for turl in comb_test_url] for i in range(EFFECTOR_SWITCH_TRIAL)]
                    HIT_dictionary[(comb, HIT_number)]['test_metas'] = [[{meta_field: scramble_combs[comb_counter][j]} for j in reversed(effector_map)] for i in range(EFFECTOR_SWITCH_TRIAL)]
                    HIT_dictionary[(comb, HIT_number)]['test_labels'] = [[scramble_combs[comb_counter][j] for j in effector_map] for i in range(EFFECTOR_SWITCH_TRIAL)]

                    # First object set 
                    HIT_dictionary[(comb, HIT_number)]['test_urls'].extend([[turl for turl in comb_test_url] for i in range(EFFECTOR_SWITCH_TRIAL)])
                    HIT_dictionary[(comb, HIT_number)]['test_metas'].extend([[{meta_field: comb[j]} for j in effector_map] for i in range(EFFECTOR_SWITCH_TRIAL)])
                    HIT_dictionary[(comb, HIT_number)]['test_labels'].extend([[comb[j] for j in effector_map] for i in range(EFFECTOR_SWITCH_TRIAL)])

                    


            else: 
                raise Exception 
            # Add 'meta_field'
            HIT_dictionary[(comb, HIT_number)]['meta_field'] = [meta_field]*n_experimental_trials




            # Lastly, prepend tutorial trials. 
            HIT_dictionary[(comb, HIT_number)]['sample_urls'] = tut_sample_urls + HIT_dictionary[(comb, HIT_number)]['sample_urls'] 
            HIT_dictionary[(comb, HIT_number)]['sample_metas'] = tut_sample_metas + HIT_dictionary[(comb, HIT_number)]['sample_metas']
            HIT_dictionary[(comb, HIT_number)]['test_urls'] = tut_test_urls + HIT_dictionary[(comb, HIT_number)]['test_urls']
            HIT_dictionary[(comb, HIT_number)]['test_metas'] = tut_test_metas + HIT_dictionary[(comb, HIT_number)]['test_metas'] 
            HIT_dictionary[(comb, HIT_number)]['test_labels'] = tut_test_labels + HIT_dictionary[(comb, HIT_number)]['test_labels']
            HIT_dictionary[(comb, HIT_number)]['meta_field'] = tut_meta_fields + HIT_dictionary[(comb, HIT_number)]['meta_field']


    # Sanity check and save
    print 'Number of HITs written:', len(HIT_dictionary.keys())

    # Write experiment_meta 
    experiment_meta = {'workers_per_HIT': number_of_workers_per_HIT, 
                        'reward_per_HIT': reward_per_HIT,
                        'target_bonus_per_HIT': target_bonus_per_HIT,
                        'nickname': experiment_nickname, 
                        'num_tutorial_trials': num_tutorial_trials, 
                        'html_src':htmlsrc, 
                        'instructions_html_src': instructions_html_src, 
                        'hit_id_save_directory': hit_id_save_directory, 
                        'experiment_param_save_directory': experiment_param_save_directory, 
                        'psychophysics_save_directory': psychophysics_save_directory, 
                        }

    if not os.path.exists(experiment_param_save_directory): os.makedirs(experiment_param_save_directory)

    save_string = os.path.join(experiment_param_save_directory, 'params_'+experiment_nickname+strftime("%Y-%m-%d--%H.%M.%S", gmtime()) + '.pk')
    pk.dump((experiment_meta, HIT_dictionary), open(save_string, "wb"))
    print 'Saved experiment parameters at:\n', save_string
    return experiment_meta, HIT_dictionary

def _make_abstractshape_tutorial_trials(num_trials = 10, nways = 2, rng_seed = 0): 
    if nways == 2: 
        test_url = [
                    'https://s3.amazonaws.com/mutatorsr/resources/response_images/left.png', 
                    'https://s3.amazonaws.com/mutatorsr/resources/response_images/right.png']
        test_url_buttons = ['https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png', 
                    'https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png']
        obj_sequencing = [0, 0, 0, 1, 1, 0, 1, 0, 1, 0] 
        effector_map = [[0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1], [0, 1], [0, 1]] # effector map for each trial
    else: 
        raise NotImplementedError

    obj_sequencing = obj_sequencing[0:num_trials]
        
    
    urls = ['https://s3.amazonaws.com/mutatorsr/resources/tutorial_assets/sample_shape1.png', 
    'https://s3.amazonaws.com/mutatorsr/resources/tutorial_assets/sample_shape2.png']
    tutorial_objects = ['shape1', 'shape2']

    tut_sample_urls, tut_sample_metas, tut_test_urls, tut_test_metas, tut_test_labels, tut_meta_fields  = [], [], [], [], [], []

    # Write down trials
    for trial_num, t in enumerate(obj_sequencing): 
        idx = t

        tut_sample_urls.append(urls[idx])
        tut_sample_metas.append({'obj': tutorial_objects[idx]})

        if trial_num < int(len(obj_sequencing)/2.): 
            # Use arrow key response images for the first half. 
            tut_test_urls.append(test_url)
        else: 
            # Switch over to white buttons halfway through. 
            tut_test_urls.append(test_url_buttons)
        
        tut_test_metas.append([{'obj': tutorial_objects[i]} for i in effector_map[trial_num]])
        tut_test_labels.append([tutorial_objects[i] for i in effector_map[trial_num]])
        tut_meta_fields.append('obj')

    # Based on tut_combs, get meta indices of images to be presented
    return tut_sample_urls, tut_sample_metas, tut_test_urls, tut_test_metas, tut_test_labels, tut_meta_fields

def _make_tutorial_trials(num_trials = 10, nways = 2, rng_seed = 0): 
    if nways == 4: 
        test_url = ['https://s3.amazonaws.com/mutatorsr/resources/response_images/up.png', 
                    'https://s3.amazonaws.com/mutatorsr/resources/response_images/left.png', 
                    'https://s3.amazonaws.com/mutatorsr/resources/response_images/right.png', 
                    'https://s3.amazonaws.com/mutatorsr/resources/response_images/down.png', ]

        test_url_buttons = ['https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png', 
                'https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png', 
                'https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png',
                'https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png'
                ]

        obj_sequencing = [0, 0, 0, 0, 1, 1, 1, 3, 2, 0] 
    if nways == 2: 
        test_url = [
                    'https://s3.amazonaws.com/mutatorsr/resources/response_images/left.png', 
                    'https://s3.amazonaws.com/mutatorsr/resources/response_images/right.png']
        test_url_buttons = ['https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png', 
                    'https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png']
        obj_sequencing = [0, 0, 0, 1, 1, 0, 1, 0, 1, 0] 
    else: 
        raise NotImplementedError

    obj_sequencing = obj_sequencing[0:num_trials]
        
    d =hvm.HvMWithDiscfade()
    meta  = d.meta
    urls = d.publish_images(range(len(d.meta)), None, 'hvm_timing', dummy_upload=True)
    tutorial_objects = ['Apricot_obj', 'motoryacht', 'z3', 'Beetle']
    meta_query = lambda x: x['var'] == 'V3'
    effector_map = range(nways)

    tut_sample_urls, tut_sample_metas, tut_test_urls, tut_test_metas, tut_test_labels, tut_meta_fields  = [], [], [], [], [], []

    # Write down trials
    for trial_num, t in enumerate(obj_sequencing): 
        obj = tutorial_objects[t]
        eligible_idx = filter(lambda x: meta_query(meta[x]) and meta[x]['obj'] == obj, range(len(meta)))
        idx = np.random.choice(eligible_idx)

        tut_sample_urls.append(urls[idx])
        tut_sample_metas.append(meta[idx])

        if trial_num < int(len(obj_sequencing)/2.): 
            # Use arrow key response images for the first half. 
            tut_test_urls.append(test_url)
        else: 
            # Switch over to white buttons halfway through. 
            tut_test_urls.append(test_url_buttons)
        
        tut_test_metas.append([{'obj': tutorial_objects[i]} for i in effector_map])
        tut_test_labels.append([tutorial_objects[i] for i in effector_map])
        tut_meta_fields.append('obj')

    # Based on tut_combs, get meta indices of images to be presented
    return tut_sample_urls, tut_sample_metas, tut_test_urls, tut_test_metas, tut_test_labels, tut_meta_fields


if __name__ == '__main__': 
    write_mturk_SRexperiment()

# SR Experiment definitions: 

# Meta containing objects / urls, and list of combinations to test
# In a HIT, number of positive trials per object in combination 
# Number of (eligible) workers per HIT

# What defines a HIT: should be accessible here 
# {images: [image_urls ... ]
# responses: [(response_image1..., position1...)]}
# correct_responses [correct_idx, ...]

# Group HITs

# Perform HITs in groups
