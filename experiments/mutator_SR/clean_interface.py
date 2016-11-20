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
	experiment_nickname = 'mut_2wayV6' # Determines outputs/[folder] in which everything will be saved
	experiment_param_save_directory = os.path.join(SAVE_DIRECTORY, experiment_nickname, 'params')
	psychophysics_save_directory = os.path.join(SAVE_DIRECTORY, experiment_nickname, 'psychophysics_data')
	hit_id_save_directory = os.path.join(SAVE_DIRECTORY, experiment_nickname, 'hit_ids/')
	
	print hit_id_save_directory


	
	# User HIT settings
	#d = mut.PilotAll_LargeSetWithDiscfade()
	#d = hvm.HvMWithDiscfade()
	d = mut.Pilot18()

	meta  = d.meta
	number_of_workers_per_HIT = 2
	number_of_HITs_per_comb = 1
	reward_per_HIT = 0.1
	target_bonus_per_HIT = 0
	nways = 2
	num_tutorial_trials = 1
	number_positive_exemplars_per_label_in_HIT = 1
	meta_query = lambda x: x['var'] == 'V6'
	meta_field = 'obj' 
	
	testable_objects = set(meta[meta_field])
	#hvm_testable_objects = [ 'bear', 'ELEPHANT_M', '_18', 'face0001', 'alfa155', 'breed_pug', 'TURTLE_L', 'Apple_Fruit_obj',  'f16', '_001', '_014', 'face0002']
	mut18_testable_objects = ['ballpile', 'basket', 'blenderball', 'blowfish', 'bouquet', 'bundle', 'city', 'dippindots', 'moth', 
	'octopus', 'pinecone', 'slug', 'spikybeast', 'staircase']

	testable_objects = mut18_testable_objects

	combs = [e for e in itertools.combinations(testable_objects, nways)]

	all_sample_urls = ['https://s3.amazonaws.com/mutatorsr/resources/mutator18_stimuli/discfaded/'+h for h in meta['id']] # in meta order
	#all_sample_urls = d.publish_images(range(len(d.meta)), None, 'hvm_timing', dummy_upload=False)

	test_url = ['https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png', 
				'https://s3.amazonaws.com/mutatorsr/resources/response_images/white_buttons/white_button512x512.png'] # up / left / right / down is the default order
	all_effector_maps = [np.random.permutation(nways) for i in range(len(combs)*number_of_HITs_per_comb)] # One map per HIT;;  indexes into comb[i]
	











	# Write HITs
	tut_sample_urls, tut_sample_metas, tut_test_urls, tut_test_metas, tut_test_labels, tut_meta_fields  = _make_tutorial_trials(num_trials = num_tutorial_trials, nways = nways, rng_seed = rng_seed); 
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
		
		# Shuffle HIT trials to mix labels temporally
		for HIT_number in range(number_of_HITs_per_comb): 
			HIT_absolute_counter = comb_counter*number_of_HITs_per_comb+HIT_number

			n_experimental_trials = len(HIT_dictionary[(comb, HIT_number)]['sample_urls'])
			HIT_trial_order = np.random.permutation(n_experimental_trials)

			# Shuffle order (currently o1, o2... contiguous). 
			HIT_dictionary[(comb, HIT_number)]['sample_urls'] = [HIT_dictionary[(comb, HIT_number)]['sample_urls'][h] for h in HIT_trial_order]
			HIT_dictionary[(comb, HIT_number)]['sample_metas'] = [HIT_dictionary[(comb, HIT_number)]['sample_metas'][h] for h in HIT_trial_order]

			# Also add test images/meta.
			effector_map = all_effector_maps[HIT_absolute_counter]
			comb_test_url = test_url # here would be the appropriate place to make alterations for m2s
			HIT_dictionary[(comb, HIT_number)]['test_urls'] = [[turl for turl in comb_test_url] for i in range(n_experimental_trials)]
			HIT_dictionary[(comb, HIT_number)]['test_metas'] = [[{meta_field: comb[j]} for j in effector_map] for i in range(n_experimental_trials)] 
			HIT_dictionary[(comb, HIT_number)]['test_labels'] = [[comb[j] for j in effector_map] for i in range(n_experimental_trials)]

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
