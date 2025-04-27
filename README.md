# calibrated-betamix

Python code of our AISTATS 2025 paper : 

    Unconditionally Calibrated Priors for Beta Mixture Density Networks
    Alix Lhéritier & Maurizio Filippone


## Instructions to reproduce the experiments 

Please see conda_cpu.yml (or conda.yml for GPU) for the dependencies.
Requires mlflow to log the results and save artifacts.

### UCI data used in the experiments

Data and splits obtained from : https://github.com/yaringal/DropoutUncertaintyExps

Please set the following variables accordingly:

	DATA_FOLDER = "../DropoutUncertaintyExps/UCI_Datasets/"
	DATASETS = ["bostonHousing", "concrete", "energy", "kin8nm", "naval-propulsion-plant", "power-plant", "protein-tertiary-structure", "wine-quality-red", "yacht"]


### UCI experiments with erf activation function:

	import numpy as np
	import os

	for SCALE in [1 ]:# ,100,10000]:
		for DATASET in DATASETS:
			splits = int(np.loadtxt(DATA_FOLDER + "/" + DATASET + "/data/n_splits.txt"))
			
			for SPLIT in range(1): #splits):
				cmd = "python code/run_betamix.py --squeeze_n 5 --n_segments 5 --activation erf --hidden_layers 1 --nodes 50 --test_M 1000 --num_chains 4 --sampler sgldAdam --sg --batch_size 256 --num_warmup 100000 --keep_every 100 --step_size .003 --prior_sigma 1. --prior_alpha_weights 1. --prior_scale_ab {} --alpha_quant 0. --seed 321 --data_folder {} --dataset {} --split {} --plot_prior --plot_orig_prior" 
				cmd = cmd.format(SCALE,DATA_FOLDER, DATASET, SPLIT)
				print(cmd)
				os.system(cmd)
		
### UCI experiments with rbf sinusoidal activation function:

	for SIGMA in [1,2,4,8]:
		for DATASET in DATASETS:
			SPLITS = int(np.loadtxt(DATA_FOLDER + "/" + DATASET + "/data/n_splits.txt"))
			
			for SPLIT in SPLITS: 
				cmd = "python code/run_betamix.py --stax --stax_parameterization standard --squeeze_n 1 --n_segments 5 --activation rbf --hidden_layers 1 --nodes 50 --test_M 1000 --num_chains 4 --library sgmcmcjax --sampler sgldAdam --sg --batch_size 256 --num_warmup 100000 --keep_every 100 --step_size .003 --prior_sigma {} --prior_alpha_weights 1. --prior_scale_ab 10000 --alpha_quant 0. --seed 321 --data_folder {} --dataset {} --split {}"
				cmd = cmd.format(SIGMA,DATA_FOLDER, DATASET, SPLIT)
				print(cmd)
				os.system(cmd)



### NYC Taxi drop-off prediction:

	Data in csv format is in the data folder of this repository (obtained from https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

	Please set the following variable accordingly:

    DATA_FOLDER = "../nycdata"

    for SPLIT in [0,1,2]:
        cmd = "python code/run_betamix.py --squeeze gaussmix --squeeze_n 74 --n_segments 83 --activation relu --stax --stax_parameterization standard_custom --hidden_layers 1 --nodes 339 --prior_alpha_weights 1. --prior_scale_ab 960.0125923836018 --prior_corr_concentration 1. --lkj cvine --logbetacdf_N 500 --num_warmup 50000 --sg --sampler sgmcmc --library blackjax --test_M 400 --num_chains 4 --step_size 0.0038375276112897197 --keep_every 10 --batch_size 80 --prior_sigma 9.50536589650343 --plot_prior --plot_prior2D --plot_final --alpha_quant 0. --seed 123 --data_folder {} --dataset NYTaxiLocationPrediction --split {}"
        cmd = cmd.format(DATA_FOLDER,SPLIT)
        print(cmd)
        os.system(cmd)

### Synthetic beta letter:

BMDN:

    from argparse import Namespace
    
	args = Namespace(n_segments=4, hidden_layers=2, nodes=50, activation='rbf', nocopula=False, nobridgebetas=False, sorted_locations=False, gmdn=False, gaussmix_with_scaling=False, library='blackjax', sampler='nuts', sg=False, noncompiled=False, maxnorm=np.inf, do_map=False, no_prior=False, freeze_hidden_after_map=False, map_epochs=10, map_step_size=None, map_batch_size=32, map_prior_sigma=1.0, vi_samples=10, test_M=400, num_warmup=1000, batch_size=80, step_size=0.0001, log_step_size=None, last_step_size=None, leap_frog_steps=None, num_chains=4, keep_every=10, prior_sigma=2.0, log_prior_sigma=None, prior_alpha_weights=1.0, prior_scale_ab=10000.0, log_prior_scale_ab=None, prior_corr_concentration=1.0, lkj='cvine', logbetacdf_N=500, nocholcorr=False, alpha_quant=0.0, seed=321, dataset='dgp', split=0, validation=False, skip_train_eval=False, squeeze_n=8, squeeze='gaussmix', n='500', plot_final=True, plot_prior=True, plot_prior2D=False, plot_orig_prior=True, maxdens=None, stax=True, stax_parameterization='standard')


    class Dict2Class(object): 
        def __init__(self, my_dict): 
            
            for key in my_dict: 
                setattr(self, key, my_dict[key]) 

    inputs = Dict2Class(args.__dict__)

    for SCALE in  [1.,100.,10000.,1000000, 100000000]:
        for SIGMA in [.25,1.,2.,4.] :

            search_space = Dict2Class({"prior_sigma":SIGMA, "prior_scale_ab": SCALE})
            cmd = f"python code/run_betamix.py --squeeze gaussmix --sampler nuts --stax --plot_final --plot_prior --plot_orig_prior --n {inputs.n} --batch_size {inputs.batch_size} --dataset {inputs.dataset} --split {inputs.split} --keep_every {inputs.keep_every} --test_M {inputs.test_M} --num_chains {inputs.num_chains} --num_warmup {inputs.num_warmup} --prior_alpha_weights {inputs.prior_alpha_weights} --seed {inputs.seed} --alpha_quant {inputs.alpha_quant} --library {inputs.library} --prior_corr_concentration {inputs.prior_corr_concentration} --lkj {inputs.lkj} --logbetacdf_N {inputs.logbetacdf_N} --stax_parameterization {inputs.stax_parameterization} --step_size {inputs.step_size} --hidden_layers {inputs.hidden_layers} --nodes {inputs.nodes} --activation {inputs.activation} --squeeze_n {inputs.squeeze_n} --n_segments {inputs.n_segments} --prior_sigma {search_space.prior_sigma} --prior_scale_ab {search_space.prior_scale_ab}"
            print(cmd)
            os.system(cmd)

GMDN:

    from argparse import Namespace
    args = Namespace(n_segments=4, hidden_layers=2, nodes=50, activation='rbf', nocopula=False, nobridgebetas=False, sorted_locations=False, gmdn=True, gaussmix_with_scaling=False, library='blackjax', sampler='nuts', sg=False, noncompiled=False, maxnorm=np.inf, do_map=False, no_prior=False, freeze_hidden_after_map=False, map_epochs=10, map_step_size=None, map_batch_size=32, map_prior_sigma=1.0, vi_samples=10, test_M=400, num_warmup=1000, batch_size=80, step_size=0.0001, log_step_size=None, last_step_size=None, leap_frog_steps=None, num_chains=4, keep_every=10, prior_sigma=2.0, log_prior_sigma=None, prior_alpha_weights=1.0, prior_scale_ab=1.0, log_prior_scale_ab=None, prior_corr_concentration=1.0, lkj='cvine', logbetacdf_N=500, nocholcorr=False, alpha_quant=0.0, seed=321, dataset='dgp', split=0, validation=False, skip_train_eval=False, squeeze_n=8, squeeze='gaussmix', n='500', plot_final=True, plot_prior=True, plot_prior2D=False, plot_orig_prior=True, maxdens=None, stax=True, stax_parameterization='standard')

    class Dict2Class(object): 
        def __init__(self, my_dict): 
            
            for key in my_dict: 
                setattr(self, key, my_dict[key]) 

    inputs = Dict2Class(args.__dict__)

    for SIGMA in [.25,1.,2.,4.]:

        search_space = Dict2Class({"prior_sigma":SIGMA})
        cmd = f"python code/run_betamix.py --squeeze gaussmix --sampler nuts --stax --plot_final --plot_prior --plot_orig_prior --n {inputs.n} --batch_size {inputs.batch_size} --dataset {inputs.dataset} --split {inputs.split} --keep_every {inputs.keep_every} --test_M {inputs.test_M} --num_chains {inputs.num_chains} --num_warmup {inputs.num_warmup} --prior_alpha_weights {inputs.prior_alpha_weights} --seed {inputs.seed} --alpha_quant {inputs.alpha_quant} --library {inputs.library} --prior_corr_concentration {inputs.prior_corr_concentration} --lkj {inputs.lkj} --logbetacdf_N {inputs.logbetacdf_N} --stax_parameterization {inputs.stax_parameterization} --step_size {inputs.step_size} --hidden_layers {inputs.hidden_layers} --nodes {inputs.nodes} --activation {inputs.activation} --squeeze_n {inputs.squeeze_n} --n_segments {inputs.n_segments} --prior_sigma {search_space.prior_sigma} --prior_scale_ab 1. --gmdn"
        print(cmd)
        os.system(cmd)
    

### NYC Taxi drop-off with small number of components:
COPULA:

    DATA_FOLDER = "../nycdata"

    from argparse import Namespace
    args = Namespace(n_segments=8, hidden_layers=2, nodes=32, activation='relu', nocopula=False, nobridgebetas=False, sorted_locations=False, gmdn=False, gaussmix_with_scaling=False, library='sgmcmcjax', sampler='sgmcmc', sg=True, noncompiled=False, maxnorm=np.inf, do_map=False, no_prior=False, freeze_hidden_after_map=False, map_epochs=10, map_step_size=None, map_batch_size=32, map_prior_sigma=1.0, vi_samples=10, test_M=400, num_warmup=50000, batch_size=80, step_size=0.003, log_step_size=None, last_step_size=None, leap_frog_steps=None, num_chains=4, keep_every=10, prior_sigma=1.0, log_prior_sigma=None, prior_alpha_weights=1.0, prior_scale_ab=1000.0, log_prior_scale_ab=None, prior_corr_concentration=10.0, lkj='cvine', logbetacdf_N=500, nocholcorr=False, alpha_quant=0.0, seed=321, dataset='NYTaxiLocationPrediction', data_folder=DATA_FOLDER, split=0, validation=False, skip_train_eval=False, squeeze_n=1, squeeze='gaussmix', n=None, plot_final=True, plot_prior=True, plot_prior2D=True, plot_orig_prior=True, maxdens=None, stax=True, stax_parameterization='standard_custom')

    class Dict2Class(object): 
        def __init__(self, my_dict): 
            
            for key in my_dict: 
                setattr(self, key, my_dict[key]) 

    inputs = Dict2Class(args.__dict__)

    for COMPONENTS in  [1,2,4,8]:
        search_space = Dict2Class({"n_segments":COMPONENTS})
        cmd = f"python code/run_betamix.py --squeeze gaussmix --sg --sampler sgmcmc --stax --plot_final --plot_prior --plot_orig_prior --plot_prior2D --batch_size {inputs.batch_size} --dataset {inputs.dataset} --split {inputs.split} --data_folder {inputs.data_folder} --keep_every {inputs.keep_every} --test_M {inputs.test_M} --num_chains {inputs.num_chains} --num_warmup {inputs.num_warmup} --prior_alpha_weights {inputs.prior_alpha_weights} --seed {inputs.seed} --alpha_quant {inputs.alpha_quant} --library {inputs.library} --prior_corr_concentration {inputs.prior_corr_concentration} --lkj {inputs.lkj} --logbetacdf_N {inputs.logbetacdf_N} --stax_parameterization {inputs.stax_parameterization} --step_size {inputs.step_size} --hidden_layers {inputs.hidden_layers} --nodes {inputs.nodes} --activation {inputs.activation} --prior_sigma {inputs.prior_sigma} --prior_scale_ab {inputs.prior_scale_ab} --squeeze_n {inputs.squeeze_n} --n_segments {search_space.n_segments}"
        print(cmd)
        os.system(cmd)

NOCOPULA :

    DATA_FOLDER = "../nycdata"

    from argparse import Namespace
    args = Namespace(n_segments=8, hidden_layers=2, nodes=32, activation='relu', nocopula=True, nobridgebetas=False, sorted_locations=False, gmdn=False, gaussmix_with_scaling=False, library='sgmcmcjax', sampler='sgmcmc', sg=True, noncompiled=False, maxnorm=np.inf, do_map=False, no_prior=False, freeze_hidden_after_map=False, map_epochs=10, map_step_size=None, map_batch_size=32, map_prior_sigma=1.0, vi_samples=10, test_M=400, num_warmup=50000, batch_size=80, step_size=0.003, log_step_size=None, last_step_size=None, leap_frog_steps=None, num_chains=4, keep_every=10, prior_sigma=1.0, log_prior_sigma=None, prior_alpha_weights=1.0, prior_scale_ab=1000.0, log_prior_scale_ab=None, prior_corr_concentration=10.0, lkj='cvine', logbetacdf_N=500, nocholcorr=False, alpha_quant=0.0, seed=321, dataset='NYTaxiLocationPrediction', data_folder=DATA_FOLDER, split=0, validation=False, skip_train_eval=False, squeeze_n=1, squeeze='gaussmix', n=None, plot_final=True, plot_prior=True, plot_prior2D=True, plot_orig_prior=True, maxdens=None, stax=True, stax_parameterization='standard_custom')

    class Dict2Class(object): 
        def __init__(self, my_dict): 
            
            for key in my_dict: 
                setattr(self, key, my_dict[key]) 

    inputs = Dict2Class(args.__dict__)

    for COMPONENTS in  [1,2,4,8]:
        search_space = Dict2Class({"n_segments":COMPONENTS})
        cmd = f"python code/run_betamix.py --squeeze gaussmix --sg --sampler sgmcmc --stax --plot_final --plot_prior --plot_orig_prior --plot_prior2D --batch_size {inputs.batch_size} --dataset {inputs.dataset} --split {inputs.split} --data_folder {inputs.data_folder} --keep_every {inputs.keep_every} --test_M {inputs.test_M} --num_chains {inputs.num_chains} --num_warmup {inputs.num_warmup} --prior_alpha_weights {inputs.prior_alpha_weights} --seed {inputs.seed} --alpha_quant {inputs.alpha_quant} --library {inputs.library} --prior_corr_concentration {inputs.prior_corr_concentration} --lkj {inputs.lkj} --logbetacdf_N {inputs.logbetacdf_N} --stax_parameterization {inputs.stax_parameterization} --step_size {inputs.step_size} --hidden_layers {inputs.hidden_layers} --nodes {inputs.nodes} --activation {inputs.activation} --prior_sigma {inputs.prior_sigma} --prior_scale_ab {inputs.prior_scale_ab} --squeeze_n {inputs.squeeze_n} --n_segments {search_space.n_segments}"
        print(cmd)
        os.system(cmd)



## License
[MIT license](https://github.com/alherit/calibrated-betamix/blob/master/LICENSE).



