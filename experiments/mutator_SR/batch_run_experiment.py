# 
import time 
import collections
import os
from __future__ import print_function
import itertools
import write_SR_experiment_parameters
import driver_mutatorSR

check_period = 60 # how often, in seconds, to check if current batch is done 
previously_seen_workers_on_obj = None; # can also load previous file here


class SerialMturkCollector:
	def __init__(self, object_combinations, Nway, previously_seen_workers_on_obj = None, sandbox = True, imageset = None): 

		self.imageset = imageset
		self.object_combinations = object_combinations
		self.Nway = Nway 
		self.previously_seen_workers_on_obj = previously_seen_workers_on_obj
		self.sandbox = sandbox

	def run(self): 

		serialized_batch_order = make_batches_from_objectlist(self.object_combinations)
		threshold_perf = 1./self.Nway
		if self.previously_seen_workers_on_obj is None: 
			print 'Did not load previously seen workers file; starting new one.'
			self.previously_seen_workers_on_obj = collections.defaultdict(list)

		# Main loop
		for batch in serialized_batch_order: 
			# First pass of HITs
			valid_assignments, all_assignments, assignments_to_repeat = self.do_batch(batch)
			self.save_psychophysics(valid_assignments, all_assignments)

			# Extend certain HITs 
			while(len(assignments_to_repeat)>0): 
				valid_repeat_assignments, all_repeat_assignments, assignments_to_repeat = self.do_repeat_tries_on_assignments(assignments_to_repeat) 
				self.save_psychophysics(valid_repeat_assignments, all_repeat_assignments)




	def do_batch(self, batch): 
		# batch is list of combs 
		batch_exps = []

		# Upload HITs for each comb. 
		for comb in batch: 
			# Write parameter file
			if(self.imageset == 'mut'): 
				params = write_SR_experiment_parameters.MutatorDiscfade_SR_Experiment(labelset = list(comb), Nway = self.Nway)
				param_path = params.save()

			elif(self.imageset == 'hvm'): 
				params = write_SR_experiment_parameters.HvM_SR_Experiment(labelset = list(comb), Nway = self.Nway)
				param_path = params.save()


			# Run python driver and get exp object
			argv = [None, param_path]
			if(self.sandbox == False): 
				argv.append('production')
			exp = driver_mutatorSR.main(argv)

			batch_exps.append(exp)


		# Check to see if HITs are done 
		batch_not_done = True 
		while(batch_not_done): 
			batch_not_done = False
			time.sleep(check_period)
			for exp in batch_exps: 
				conn = exp.conn # boto.mturk.connection.MTurkConnection
				hitids = exp.hitids 

				for hit_id in hitids: 
					result_set = conn.get_assignments(hit_id)
					num_results = result_set['NumResults']

					if(num_results < exp.max_assignments): 
						batch_not_done = True

		# The batch is done
		for exp in batch_exps: 
			exp.updateDBwithHITs(exp.hitids)
        

        pk.dump(exp.all_data, open(('./result_pickles/results_'+exp_spec['nickname_prefix']+argv[2]), "wb"))



		return valid_assignments, all_assignments, assignments_to_repeat

	def save_psychophysics(self, valid_assignments, all_assignments): 
		return

	def do_repeat_tries_on_assignments(self, assignments_to_repeat): 
		# assignments_to_repeat: {HITID:numrepeats}
		return


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



def main(argv = []): 
	# Purpose: to group HITs together into batches that are ok to do all-of-them in. Separate batches temporally to reduce the risk of repeat workers, 
	# but if there are repeat workers, then do not pay the worker and extend the HIT. 

	imageset = 'mut'
	sandbox = True
	Nway = 4

	objectlist = [ 'bear', 'ELEPHANT_M', '_18', 'face0001', 'alfa155', 'breed_pug', 'TURTLE_L', 'Apple_Fruit_obj',  'f16', '_001', '_014', 'face0002']
	objectlist = [#'algae',
                 'ballpile',
                 'blenderball',
                 'blowfish',
                 'cinnamon',
                 #'city',
                 'dippindots',
                 'dirtyball',
                 #'octopus',
                 'spikybeast',
                 'staircase',
                 #'twoshells'
                 ]

	object_combinations =  [objectlist for e in itertools.combinations(objectlist, Nway)]  

	s = SerialMturkCollector(object_combinations, Nway, sandbox = sandbox, imageset = imageset)
	s.run()


if __name__ == '__main__':
    main(sys.argv)

