from __future__ import print_function

import sys 
import collections
import numpy as np 
import cPickle as pk
import copy
import os 
from time import gmtime, strftime
from tqdm import tqdm
import time 

#import mturkutils.base as base 
WORKER_DUMPINGGROUNDS_PATH = './worker_archive'
CHECK_PERIOD = 10 # in seconds 
WORKER_DICTIONARY_FILEPATH = None # {Worker* / IP*  : objects previously observed done}
if(WORKER_DICTIONARY_FILEPATH == None): 
    worker_dictionary = collections.defaultdict(list)
else: 
    worker_dictionary = pk.load(open(WORKER_DICTIONARY_FILEPATH, 'rb'))

def main(argv = [], debug = False): 
    if(len(argv)>2): 
        if(argv[2] == 'production'): production == True
    else: production = False

    experiment_parameters_path = argv[1]
    experiment_parameters = pk.load(open(experiment_parameters_path, 'rb'))

    experiment_meta = experiment_parameters[0]
    HIT_dictionary = experiment_parameters[1]

    SAVE_DIRECTORY = experiment_meta['experiment_param_save_directory']

    confusions = [k[0] for k in HIT_dictionary.keys()]
    groups = make_batches_from_objectlist(confusions)
    
    for i, g in enumerate(groups.keys()): 
        print('Group', i, '. # confusions:', len(groups[g]), '. Group members:', groups[g])


    comb_HIT_repetitions = list(set([k[1] for k in HIT_dictionary.keys()])) # multiple HITS per combination, then run this batch round-robin. 

    ## Main state loop 
    if debug: 
        return 

    for h in comb_HIT_repetitions: 
        print('On set', h+1, 'out of', len(comb_HIT_repetitions), 'sets.')

        for g in groups.keys(): 
            

            # Print group information 
            print('*************** On group', str(g+1), 'of', len(groups.keys()), ' ***************')
            print('# confusions:', len(groups[g]), '\n')
            print('Group members:\n',)
            for member in groups[g]:  print(member)

            # Write to disk the group's experiment parameters
            current_HIT_repetition = h
            current_combs = groups[g]

            group_HIT_dictionary_keys = [k for k in HIT_dictionary.keys() if k[0] in current_combs and k[1] == current_HIT_repetition]
            
            experiment_meta = experiment_meta
            group_HIT_dictionary = {key: value for (key, value) in zip(group_HIT_dictionary_keys, [HIT_dictionary[k] for k in group_HIT_dictionary_keys])}

            group_parameters_save_path = os.path.join(SAVE_DIRECTORY, 'group'+str(g)+'_set'+str(h)+'_params_'+strftime("%Y-%m-%d--%H.%M.%S", gmtime()) + '.pk')
            pk.dump((experiment_meta, group_HIT_dictionary), open(group_parameters_save_path, "wb"))
            

            # Make HTMLs 
            import driverClean
            if(production): 
                exp = driverClean.main(['', group_parameters_save_path, 'production'])
            else: 
                exp = driverClean.main(['', group_parameters_save_path])

            #time.sleep(120) # wait for HITs to be uploaded 

            # State loop: monitor HITs until they are all finished with eligible workers. 
            target_number_of_assignments = {hid: experiment_meta['workers_per_HIT'] for hid in exp.hitids} # Update based on extended HITs

            group_done = False
            hit_statuses = [False for i in range(len(exp.hitids))]
            reviewed_assignments = [] # review assignments only once ever 

            group_start_time = time.time()
            while not group_done: 
                time.sleep(CHECK_PERIOD)
                print('Group elapsed time:', time.time() - group_start_time, 'sec', end = '.')
                print('HIT completion status(es):', [[h__id, hs] for h__id, hs in zip(exp.hitids, hit_statuses)], end = '\r')
                if np.all(hit_statuses): 
                    group_done = True
                    print('All HITs were completed.')


                for hit_status_index, one_hit_id in enumerate(exp.hitids):
                    data = exp.getHITdata(one_hit_id, verbose = False)
                    number_of_completed_assignments = len(data)


                    if number_of_completed_assignments == target_number_of_assignments[one_hit_id]: 
                        HIT_done = True # Unless this HIT is extended, below.

                        for one_assignment in data:  
                            if one_assignment in reviewed_assignments: 
                                # Already reviewed this one. 
                                continue 
                            else: 
                                reviewed_assignments.append(one_assignment)

                            error_rate = one_assignment['Error']
                            worker_id = one_assignment['WorkerID']
                            ip_address = one_assignment['IPaddress']
                            assignment_id = one_assignment['AssignmentID']

                            print('Worker ID:', worker_id)
                            print('IP address:', ip_address)
                            print('Performance:', error_rate)
                            
                            # If this worker or ipaddress has seen any experimental objects that are in the current HIT, extend the HIT by one
                            experimental_trial_indices = filter(lambda x: one_assignment['TrialType'][x] == 'experimental', range(len(one_assignment['TrialType'])))
                            objects = [x['Sample']['obj'] for x in one_assignment['ImgData']]
                            experimental_objects = set([objects[x] for x in experimental_trial_indices])

                            extend_HIT = False
                            if len(set(worker_dictionary[worker_id]).intersection(experimental_objects)) > 0: 
                                extend_HIT = True 
                                print('The worker', worker_id, 'has seen objects in this HIT.')
                                
                            if len(set(worker_dictionary[ip_address]).intersection(experimental_objects)) > 0: 
                                extend_HIT = False 
                                print('The IP address', ip_address, 'has seen objects in this HIT.')

                            # Reject assignment based on performance
                            chance_error_rate = 1. - 1./len(one_assignment['ImgData'][0]['Test'])
                            
                            if(error_rate > chance_error_rate): 
                                exp.conn.reject_assignment(assignment_id, feedback = 'The error rate was greater than chance guessing.')
                                print('The chance error rate is', chance_error_rate)
                                print('The assignment', assignment_id, 'had a performance of', 1-error_rate, 'and did not meet the minimum of', 1-chance_error_rate)
                                extend_HIT = True


                            # If the worker has either 1) seen these objects before or 2) had chance performance, extend HIT. 
                            if(extend_HIT == True): 
                                print('Extending HIT by 1.')
                                exp.conn.extend_hit(one_hit_id, assignments_increment = 1)
                                target_number_of_assignments[one_hit_id]+=1
                                HIT_done = False


                            # Record that this worker and ipaddress has seen current objects
                            worker_dictionary[worker_id].extend(list(experimental_objects))
                            worker_dictionary[ip_address].extend(list(experimental_objects))

                            #worker_dictionary[worker_id] = list(set(worker_dictionary[worker_id])) # remove duplicate objects.
                            #worker_dictionary[ip_address] = list(set(worker_dictionary[ip_address]))

                            
                        # If no extensions were made in any assignments, then this HIT is done. 
                        if(HIT_done): 
                            hit_statuses[hit_status_index] = True

            # The group is done. Push all results to database / a local pickle file. 
            exp.updateDBwithHITs(exp.hitids)

            psychophysics_save_dir = os.path.join(SAVE_DIRECTORY, 'batch_psychophysics_data')
            if not os.path.exists(psychophysics_save_dir): os.makedirs(psychophysics_save_dir)
            pk.dump(exp.all_data, 
                open(os.path.join(psychophysics_save_dir, 'results_'+strftime("%Y-%m-%d--%H.%M.%S", gmtime()))
            , "wb"))

    # Save worker dictionary 
    pk.dump(worker_dictionary, open(os.path.join(WORKER_DUMPINGGROUNDS_PATH, 'worker_dictionary'+experiment_meta['nickname']+strftime("%Y-%m-%d--%H.%M.%S", gmtime()), "wb")))
                



def make_batches_from_objectlist(confusions):
    # Makes non-overlapping groups of combs (list of lists)

    groups = collections.defaultdict(list)
    
    for c in confusions: 
        group_suitor = 0
        no_group_home = True
        
        while(no_group_home):
            if np.any([obj in conf for obj in c for conf in groups[group_suitor]]): 
                
                group_suitor+=1
            else: 
                groups[group_suitor].append(c)
                no_group_home = False

    return groups



if __name__ == '__main__': 
    # Commandline syntax: python clean_batchrun.py [paramsfilepath] 
    main(sys.argv)
