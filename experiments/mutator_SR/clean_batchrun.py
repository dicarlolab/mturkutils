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
import datetime
import json

import boto.mturk.connection

#import mturkutils.base as base 
WORKER_DUMPINGGROUNDS_PATH = './worker_archive'
LOG_DUMPINGGROUNDS_PATH = './batch_logs'
CHECK_PERIOD = 10 # in seconds 
INTERGROUP_PERIOD = 300 # in seconds 

def main(argv = [], debug = False): 
    WORKER_DICTIONARY_FILEPATH = '/mindhive/dicarlolab/u/mil/mtk_utator_SR/worker_archive/worker_dict_2wayV6.txt'

    production = False
    if(len(argv)>2): 
        if(argv[2] == 'production'): 
            print('Running batch in production mode.')
            production = True
    else: 
        print('Running batch in sandbox mode.')
        production = False
    experiment_parameters_path = argv[1]
    experiment_parameters = pk.load(open(experiment_parameters_path, 'rb'))

    experiment_meta = experiment_parameters[0]
    HIT_dictionary = experiment_parameters[1]

    PARAM_SAVE_DIRECTORY = experiment_meta['experiment_param_save_directory']
    PSYCHOPHYSICS_SAVE_DIRECTORY = experiment_meta['psychophysics_save_directory']

    maxed_out_HITs_to_repeat = []
    LOG_FILE_JSON_FILEPATH = os.path.join(LOG_DUMPINGGROUNDS_PATH, 'logs_batchrun_'+experiment_meta['nickname']+'_'+strftime("%Y.%m.%d-%H:%M:%S", gmtime()) + '.txt')

    # Load / make worker dictionary. 
    
    if WORKER_DICTIONARY_FILEPATH == None: 
        worker_dictionary = collections.defaultdict(lambda: collections.defaultdict(list))
        if not os.path.exists(WORKER_DUMPINGGROUNDS_PATH): os.makedirs(WORKER_DUMPINGGROUNDS_PATH)
        WORKER_DICTIONARY_FILEPATH = os.path.join(WORKER_DUMPINGGROUNDS_PATH, 'worker_dictionary_'+experiment_meta['nickname']+'.txt')

        if os.path.isfile(WORKER_DICTIONARY_FILEPATH): 
            print('A worker dictionary already exists at', WORKER_DICTIONARY_FILEPATH, '. Use this one, or delete it and re-run.')
            raise Exception 

        print('Writing new worker dictionary at', WORKER_DICTIONARY_FILEPATH)
        save_worker_dict(worker_dictionary, WORKER_DICTIONARY_FILEPATH)

    else: 
        print('Using previous dictionary at', WORKER_DICTIONARY_FILEPATH)
        worker_dictionary = load_worker_dict_from_file(WORKER_DICTIONARY_FILEPATH)
        


    # Start making groups
    confusions = [k[0] for k in HIT_dictionary.keys()]
    groups = make_batches_from_objectlist(confusions)
    
    for i, g in enumerate(groups.keys()): 
        print('Group', i, '. # confusions:', len(groups[g]), '. Group members:', groups[g])


    comb_HIT_repetitions = list(set([k[1] for k in HIT_dictionary.keys()])) # multiple HITS per combination, then run this batch round-robin. 

    ## Main state loop 
    if debug: 
        return 

    batch_start_time = time.time()
    for h in comb_HIT_repetitions: 
        print('On set', h+1, 'out of', len(comb_HIT_repetitions), 'sets.')
        
        for g in groups.keys(): 
            

            # Print group information 
            print('\n\n*************** On group', str(g+1), 'of', len(groups.keys()), ' ***************')
            print('# confusions:', len(groups[g]), '\n')
            print('Group members:\n',)
            for member in groups[g]:  print(member)

            # Write to disk the group's experiment parameters
            current_HIT_repetition = h
            current_combs = groups[g]

            group_HIT_dictionary_keys = [k for k in HIT_dictionary.keys() if k[0] in current_combs and k[1] == current_HIT_repetition]
            
            experiment_meta = experiment_meta
            group_HIT_dictionary = {key: value for (key, value) in zip(group_HIT_dictionary_keys, [HIT_dictionary[k] for k in group_HIT_dictionary_keys])}

            group_parameters_save_path = os.path.join(PARAM_SAVE_DIRECTORY, 'group'+str(g)+'_set'+str(h)+'_params_'+strftime("%Y-%m-%d--%H.%M.%S", gmtime()) + '.pk')
            pk.dump((experiment_meta, group_HIT_dictionary), open(group_parameters_save_path, "wb"))
            

            # Make HTMLs 
            import driverClean
            if(production): 
                exp = driverClean.main(['', group_parameters_save_path, 'production'])
            else: 
                exp = driverClean.main(['', group_parameters_save_path])


            # State loop: monitor HITs until they are all finished with eligible workers. 
            target_number_of_assignments = {hid: experiment_meta['workers_per_HIT'] for hid in exp.hitids} # Update based on extended HITs

            group_done = False
            hit_statuses = {hid: False for hid in exp.hitids}
            hit_n_completions = {hid: 0 for hid in exp.hitids}

            reviewed_assignments = [] # review assignments only once ever 

            group_start_time = time.time()
            while not group_done: 
                time.sleep(CHECK_PERIOD)
                print('%.0fs of'%(time.time() - group_start_time), '%.0fs total elapsed'%(time.time() - batch_start_time), end = '. ')

                print('HIT completion status(es):', [[str(hit_n_completions[h___id])+'/'+str(target_number_of_assignments[h___id])] for h___id in exp.hitids])
                sys.stdout.flush()
                if np.all([hit_statuses[h___id] for h___id in hit_statuses.keys()]): 
                    print(hit_statuses)
                    group_done = True
                    print('\nAll HITs were completed.')


                for one_hit_id in exp.hitids:
                    data = exp.getHITdata(one_hit_id, verbose = False)
                    number_of_completed_assignments = len(data)
                    hit_n_completions[one_hit_id] = number_of_completed_assignments

            
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

                        print('\nWorker ID:', worker_id)
                        print('IP address:', ip_address)
                        print('Performance:', 1.-error_rate)
                        
                        # If this worker or ipaddress has seen any experimental objects that are in the current HIT, extend the HIT by one
                        experimental_trial_indices = filter(lambda x: one_assignment['TrialType'][x] == 'experimental', range(len(one_assignment['TrialType'])))
                        objects = [x['Sample']['obj'] for x in one_assignment['ImgData']]
                        experimental_objects = set([objects[x] for x in experimental_trial_indices])

                        extend_HIT = False

                        worker_dictionary = load_worker_dict_from_file(WORKER_DICTIONARY_FILEPATH)
                        workerID_previously_seen_objects = worker_dictionary[worker_id].keys()
                        IP_previously_seen_objects = worker_dictionary[ip_address].keys()

                        if len(set(workerID_previously_seen_objects).intersection(experimental_objects)) > 0: 
                            extend_HIT = True 
                            print('The worker', worker_id, 'has seen objects in this HIT.')

                            if(worker_id == 'A2J5FDWXI0FX8O' or worker_id == 'A38MRYHPG90S2N'): 
                                extend_HIT = False
                                print('But it\'s my HITID.')
                            
                        if len(set(IP_previously_seen_objects).intersection(experimental_objects)) > 0: 
                            extend_HIT = True 
                            print('The IP address', ip_address, 'has seen objects in this HIT.')

                            if(ip_address == '18.93.15.127'): 
                                extend_HIT = False
                                print('But it\'s my IP address.')
                        # Reject assignment based on performance
                        chance_error_rate = 1. - 1./len(one_assignment['ImgData'][0]['Test'])
                        
                        if(error_rate >= chance_error_rate): 
                            exp.conn.reject_assignment(assignment_id, feedback = 'The error rate was greater than chance guessing.')
                            print('The chance error rate is', chance_error_rate)
                            print('The assignment', assignment_id, 'had a performance of', 1-error_rate, 'and did not exceed the minimum of', 1-chance_error_rate)
                            print('Rejected the assignment', assignment_id)
                            extend_HIT = True
                        else: 
                            # Approve anyone who scored above threshold 
                            exp.conn.approve_assignment(assignment_id)

                            # And pay rightful bonus, if applicable
                            if(experiment_meta['target_bonus_per_HIT'] > 0): 
                                from boto.mturk.price import Price as Price
                                performance_bonus = float("%.2f" % one_assignment['Bonus'])
                                max_bonus = 0.2
                                bonus_amount = min(max_bonus, performance_bonus)
                                Price_Object = Price(amount = bonus_amount)
                                reason_string = 'This is your performance-based bonus. Thank you for helping us in our goal of understanding the brain. - the DiCarlo Lab at MIT'
                                exp.conn.grant_bonus(worker_id, assignment_id, Price_Object, reason_string)
                                print('Paid bonus: ', bonus_amount)

                            print('Approved', worker_id, 'on assignment', assignment_id)
                            

                        # If the worker has either 1) seen these objects before or 2) had chance performance, extend HIT. 
                        if(extend_HIT == True): 
                            try: 
                                print('Extending HIT by 1 to', target_number_of_assignments[one_hit_id]+1)
                                exp.conn.extend_hit(one_hit_id, assignments_increment = 1)
                                target_number_of_assignments[one_hit_id]+=1 
                                # an error is thrown if the initial number of workers was less than 10, and one is extending to geq 10. 
                                # http://docs.aws.amazon.com/AWSMechTurk/latest/AWSMturkAPI/ApiReference_ExtendHITOperation.html
                            except boto.mturk.connection.MTurkRequestError: 
                                print('Cannot extend this HIT past 10 by design of AWS API. Calling it a lost cause...')
                                maxed_out_HITs_to_repeat.append(one_hit_id)
                                save_log(maxed_out_HITs_to_repeat, LOG_FILE_JSON_FILEPATH)
                                
                        # Record that this worker and ipaddress has seen current objects, and the timestamps of exposures. 
                        string_experiment_start_time = one_assignment['TimingInfo']['ExpLoadTime']
                        experiment_start_time = time.mktime(datetime.datetime.strptime(string_experiment_start_time, "%Y-%m-%dT%H:%M:%S.%fZ").timetuple())
                        trial_times_since_experiment_start = np.array(one_assignment['TimingInfo']['TrialStartTimes'])
                        trial_timestamps = list(experiment_start_time - trial_times_since_experiment_start)

                        for e_obj in experimental_objects: 
                            e_obj_trial_indices = filter(lambda x: one_assignment['ImgData'][x]['Sample']['obj'] == e_obj, range(len(one_assignment['ImgData'])))
                            e_obj_timestamps = [trial_timestamps[tidx] for tidx in e_obj_trial_indices]
                            worker_dictionary[worker_id][e_obj].extend(e_obj_timestamps)
                            worker_dictionary[ip_address][e_obj].extend(e_obj_timestamps)

                        # Update worker dictionary with this assignment.
                        save_worker_dict(worker_dictionary, WORKER_DICTIONARY_FILEPATH)
                        

                    if(number_of_completed_assignments >= target_number_of_assignments[one_hit_id]): 
                        HIT_done = True 
                    if(number_of_completed_assignments < target_number_of_assignments[one_hit_id]): 
                        HIT_done = False 

                    # If no extensions were made in any assignments, then this HIT is done. 
                    if(HIT_done): 
                        #print('HIT completed.')
                        hit_statuses[one_hit_id] = True

            # The group is done. Push all results to database / a local pickle file. 
            print('Pushing to database...')
            exp.updateDBwithHITs(exp.hitids)

            if not os.path.exists(PSYCHOPHYSICS_SAVE_DIRECTORY): os.makedirs(PSYCHOPHYSICS_SAVE_DIRECTORY)
            group_file_savepath = os.path.join(PSYCHOPHYSICS_SAVE_DIRECTORY, 'group'+str(g)+'_set'+str(h)+'_psychophysics_'+strftime("%Y-%m-%d--%H.%M.%S", gmtime()))
            pk.dump(exp.all_data, open(group_file_savepath, "wb"))
            print('Saved group data at', group_file_savepath)

        time.sleep(INTERGROUP_PERIOD)
    
    # Save worker dictionary one last time 
    save_worker_dict(worker_dictionary, WORKER_DICTIONARY_FILEPATH)

def save_log(log, savepath): 
    with open(savepath, 'w') as fp:    
        json.dump(log, fp)
def open_log(filepath): 
    with open(filepath) as data_file: 
        log = json.load(data_file)
    return log

def save_worker_dict(worker_dictionary, savepath): 
    with open(savepath, 'w') as fp:    
        json.dump(worker_dictionary, fp)

def load_worker_dict_from_file(filepath): 
    # Loads JSON and turns it into 
    # a defaultdict(lambda x: defaultdict(list))

    retry = 0
    while retry < 10: 
        try: 
            with open(filepath) as data_file: 
                worker_dictionary_raw = json.load(data_file)
            retry = 10
        except: 
            retry+=1
            print('Retrying write.', retry)
            time.sleep(2)
            continue 


    worker_dictionary = collections.defaultdict(lambda: collections.defaultdict(list), worker_dictionary_raw)

    for key in worker_dictionary.keys(): 
        worker_dictionary[key] = collections.defaultdict(list, worker_dictionary[key])

    return worker_dictionary 


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
