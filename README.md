# calibrated-betamix

Python code of our AISTATS 2025 paper : 

    Unconditionally Calibrated Priors for Beta Mixture Density Networks
    Alix Lhéritier & Maurizio Filippone


## Instructions to reproduce the experiments 

Please see conda_cpu.yml for the dependencies.

### UCI experiments with erf activation function:

    for SCALE in [1,100,10000]:
    	for DATASET in datasets:
    		splits = int(np.loadtxt(_DATA_DIRECTORY_PATH + "/" + dataset + "/data/n_splits.txt"))
    		
    		for SPLIT in splits: 
    			python run_betamix.py --squeeze_n 5 --n_segments 5 --activation erf --hidden_layers 1 --nodes 50 --test_M 1000 --num_chains 4 --sampler sgldAdam --sg --batch_size 256 --num_warmup 100000 --keep_every 100 --step_size .003 --prior_sigma 1. --prior_alpha_weights 1. --prior_scale_ab SCALE --alpha_quant 0. --seed 321 --data_folder DATA_FOLDER --dataset DATASET --split SPLIT --plot_prior --plot_orig_prior 
		
### UCI experiments with rbf sinusoidal activation function:

    for SIGMA in [1,2,4,8]:
    	for DATASET in DATASETS:
    		splits = int(np.loadtxt(_DATA_DIRECTORY_PATH + "/" + dataset + "/data/n_splits.txt"))
    		
    		for SPLITin splits: 
    			python run_betamix.py --stax --stax_parameterization standard --squeeze_n 1 --n_segments 5 --activation rbf --hidden_layers 1 --nodes 50 --test_M 1000 --num_chains 4 --library sgmcmcjax --sampler sgldAdam --sg --batch_size 256 --num_warmup 100000 --keep_every 100 --step_size .003 --prior_sigma SIGMA --prior_alpha_weights 1. --prior_scale_ab 10000 --alpha_quant 0. --seed 321 --data_folder DATA_FOLDER --dataset DATASET --split SPLIT

### NYC Taxi drop-off prediction:

    for SPLIT in [0,1,2]:
    	python run_betamix.py --squeeze gaussmix --squeeze_n 74 --n_segments 83 --activation relu --stax --stax_parameterization standard_custom --hidden_layers 1 --nodes 339 --prior_alpha_weights 1. --prior_scale_ab 960.0125923836018 --prior_corr_concentration 1. --lkj cvine --logbetacdf_N 500 --num_warmup 50000 --sg --sampler sgmcmc --library blackjax --test_M 400 --num_chains 4 --step_size 0.0038375276112897197 --keep_every 10 --batch_size 80 --prior_sigma 9.50536589650343 --plot_prior --plot_prior2D --plot_final --alpha_quant 0. --seed 123 --data_folder DATA_FOLDER --dataset NYTaxiLocationPrediction --split SPLIT

### Synthetic beta letter:

BMDN FIXED PARAMETERS:

    Namespace(n_segments=4, hidden_layers=2, nodes=50, activation='rbf', nocopula=False, nobridgebetas=False, sorted_locations=False, gmdn=False, gaussmix_with_scaling=False, library='blackjax', sampler='nuts', sg=False, noncompiled=False, maxnorm=inf, do_map=False, no_prior=False, freeze_hidden_after_map=False, map_epochs=10, map_step_size=None, map_batch_size=32, map_prior_sigma=1.0, vi_samples=10, test_M=400, num_warmup=1000, batch_size=80, step_size=0.0001, log_step_size=None, last_step_size=None, leap_frog_steps=None, num_chains=4, keep_every=10, prior_sigma=2.0, log_prior_sigma=None, prior_alpha_weights=1.0, prior_scale_ab=10000.0, log_prior_scale_ab=None, prior_corr_concentration=1.0, lkj='cvine', logbetacdf_N=500, nocholcorr=False, alpha_quant=0.0, seed=321, dataset='dgp', data_folder='azureml://subscriptions/562ec300-1b5b-4660-9f11-8ec505c1b478/resourcegroups/alix_resource_group/workspaces/cde/datastores/workspaceblobstore/paths/LocalUpload/5ea147406ff14f86e0adc51d227f3a65/data/', split=0, validation=False, skip_train_eval=False, squeeze_n=8, squeeze='gaussmix', n='500', plot_final=True, plot_prior=True, plot_prior2D=False, plot_orig_prior=True, maxdens=None, stax=True, stax_parameterization='standard')

BMDN VARYING PARAMETERS:

    for SCALE in  [.1, 1.,10.,100.,1000.]
    	for SIGMA in [.25,.5, 1.,2.,4.] :
    		python run_betamix.py --squeeze gaussmix --sampler nuts --stax --plot_final --plot_prior --plot_orig_prior --n ${{inputs.n}} --batch_size ${{inputs.batch_size}} --dataset ${{inputs.dataset}} --split ${{inputs.split}} --data_folder ${{inputs.data_folder}} --keep_every ${{inputs.keep_every}} --test_M ${{inputs.test_M}} --num_chains ${{inputs.num_chains}} --num_warmup ${{inputs.num_warmup}} --prior_alpha_weights ${{inputs.prior_alpha_weights}} --seed ${{inputs.seed}} --alpha_quant ${{inputs.alpha_quant}} --library ${{inputs.library}} --prior_corr_concentration ${{inputs.prior_corr_concentration}} --lkj ${{inputs.lkj}} --logbetacdf_N ${{inputs.logbetacdf_N}} --stax_parameterization ${{inputs.stax_parameterization}} --step_size ${{inputs.step_size}} --hidden_layers ${{inputs.hidden_layers}} --nodes ${{inputs.nodes}} --activation ${{inputs.activation}} --squeeze_n ${{inputs.squeeze_n}} --n_segments ${{inputs.n_segments}} --prior_sigma ${{search_space.prior_sigma}} --prior_scale_ab ${{search_space.prior_scale_ab}}

GMDN FIXED PARAMETERS:

    Namespace(n_segments=4, hidden_layers=2, nodes=50, activation='rbf', nocopula=False, nobridgebetas=False, sorted_locations=False, gmdn=True, gaussmix_with_scaling=False, library='blackjax', sampler='nuts', sg=False, noncompiled=False, maxnorm=inf, do_map=False, no_prior=False, freeze_hidden_after_map=False, map_epochs=10, map_step_size=None, map_batch_size=32, map_prior_sigma=1.0, vi_samples=10, test_M=400, num_warmup=1000, batch_size=80, step_size=0.0001, log_step_size=None, last_step_size=None, leap_frog_steps=None, num_chains=4, keep_every=10, prior_sigma=2.0, log_prior_sigma=None, prior_alpha_weights=1.0, prior_scale_ab=1.0, log_prior_scale_ab=None, prior_corr_concentration=1.0, lkj='cvine', logbetacdf_N=500, nocholcorr=False, alpha_quant=0.0, seed=321, dataset='dgp', data_folder='azureml://subscriptions/562ec300-1b5b-4660-9f11-8ec505c1b478/resourcegroups/alix_resource_group/workspaces/cde/datastores/workspaceblobstore/paths/LocalUpload/5ea147406ff14f86e0adc51d227f3a65/data/', split=0, validation=False, skip_train_eval=False, squeeze_n=8, squeeze='gaussmix', n='500', plot_final=True, plot_prior=True, plot_prior2D=False, plot_orig_prior=True, maxdens=None, stax=True, stax_parameterization='standard')


GMDN VARYING PARAMETERS:

    for SIGMA in [.25,.5, 1.,2.,4.] :
    	python run_betamix.py --squeeze gaussmix --sampler nuts --stax --plot_final --plot_prior --plot_orig_prior --n ${{inputs.n}} --batch_size ${{inputs.batch_size}} --dataset ${{inputs.dataset}} --split ${{inputs.split}} --data_folder ${{inputs.data_folder}} --keep_every ${{inputs.keep_every}} --test_M ${{inputs.test_M}} --num_chains ${{inputs.num_chains}} --num_warmup ${{inputs.num_warmup}} --prior_alpha_weights ${{inputs.prior_alpha_weights}} --seed ${{inputs.seed}} --alpha_quant ${{inputs.alpha_quant}} --library ${{inputs.library}} --prior_corr_concentration ${{inputs.prior_corr_concentration}} --lkj ${{inputs.lkj}} --logbetacdf_N ${{inputs.logbetacdf_N}} --stax_parameterization ${{inputs.stax_parameterization}} --step_size ${{inputs.step_size}} --hidden_layers ${{inputs.hidden_layers}} --nodes ${{inputs.nodes}} --activation ${{inputs.activation}} --squeeze_n ${{inputs.squeeze_n}} --n_segments ${{inputs.n_segments}} --prior_sigma ${{search_space.prior_sigma}} --prior_scale_ab 1. --gmdn
    

### NYC Taxi drop-off with small number of components:
COPULA FIXED PARAMETERS:

    Namespace(n_segments=8, hidden_layers=2, nodes=32, activation='relu', nocopula=False, nobridgebetas=False, sorted_locations=False, gmdn=False, gaussmix_with_scaling=False, library='sgmcmcjax', sampler='sgmcmc', sg=True, noncompiled=False, maxnorm=inf, do_map=False, no_prior=False, freeze_hidden_after_map=False, map_epochs=10, map_step_size=None, map_batch_size=32, map_prior_sigma=1.0, vi_samples=10, test_M=400, num_warmup=50000, batch_size=80, step_size=0.003, log_step_size=None, last_step_size=None, leap_frog_steps=None, num_chains=4, keep_every=10, prior_sigma=1.0, log_prior_sigma=None, prior_alpha_weights=1.0, prior_scale_ab=1000.0, log_prior_scale_ab=None, prior_corr_concentration=10.0, lkj='cvine', logbetacdf_N=500, nocholcorr=False, alpha_quant=0.0, seed=321, dataset='NYTaxiLocationPrediction', data_folder='azureml://subscriptions/ff8eedcc-6219-4b62-b3a1-018023bc227c/resourcegroups/alix-rg/workspaces/cde/datastores/workspaceblobstore/paths/LocalUpload/5ea147406ff14f86e0adc51d227f3a65/data/', split=0, validation=False, skip_train_eval=False, squeeze_n=1, squeeze='gaussmix', n=None, plot_final=True, plot_prior=True, plot_prior2D=True, plot_orig_prior=True, maxdens=None, stax=True, stax_parameterization='standard_custom')

NOCOPULA FIXED PARAMETERS:

    Namespace(n_segments=8, hidden_layers=2, nodes=32, activation='relu', nocopula=True, nobridgebetas=False, sorted_locations=False, gmdn=False, gaussmix_with_scaling=False, library='sgmcmcjax', sampler='sgmcmc', sg=True, noncompiled=False, maxnorm=inf, do_map=False, no_prior=False, freeze_hidden_after_map=False, map_epochs=10, map_step_size=None, map_batch_size=32, map_prior_sigma=1.0, vi_samples=10, test_M=400, num_warmup=50000, batch_size=80, step_size=0.003, log_step_size=None, last_step_size=None, leap_frog_steps=None, num_chains=4, keep_every=10, prior_sigma=1.0, log_prior_sigma=None, prior_alpha_weights=1.0, prior_scale_ab=1000.0, log_prior_scale_ab=None, prior_corr_concentration=10.0, lkj='cvine', logbetacdf_N=500, nocholcorr=False, alpha_quant=0.0, seed=321, dataset='NYTaxiLocationPrediction', data_folder='azureml://subscriptions/ff8eedcc-6219-4b62-b3a1-018023bc227c/resourcegroups/alix-rg/workspaces/cde/datastores/workspaceblobstore/paths/LocalUpload/5ea147406ff14f86e0adc51d227f3a65/data/', split=0, validation=False, skip_train_eval=False, squeeze_n=1, squeeze='gaussmix', n=None, plot_final=True, plot_prior=True, plot_prior2D=True, plot_orig_prior=True, maxdens=None, stax=True, stax_parameterization='standard_custom')

VARYING COMPONENTS:

    for COMPONENTS in  [1,2,4,8]  
    	python run_betamix.py --squeeze gaussmix --sg --sampler sgmcmc --stax --plot_final --plot_prior --plot_orig_prior --plot_prior2D --batch_size ${{inputs.batch_size}} --dataset ${{inputs.dataset}} --split ${{inputs.split}} --data_folder ${{inputs.data_folder}} --keep_every ${{inputs.keep_every}} --test_M ${{inputs.test_M}} --num_chains ${{inputs.num_chains}} --num_warmup ${{inputs.num_warmup}} --prior_alpha_weights ${{inputs.prior_alpha_weights}} --seed ${{inputs.seed}} --alpha_quant ${{inputs.alpha_quant}} --library ${{inputs.library}} --prior_corr_concentration ${{inputs.prior_corr_concentration}} --lkj ${{inputs.lkj}} --logbetacdf_N ${{inputs.logbetacdf_N}} --stax_parameterization ${{inputs.stax_parameterization}} --step_size ${{inputs.step_size}} --hidden_layers ${{inputs.hidden_layers}} --nodes ${{inputs.nodes}} --activation ${{inputs.activation}} --prior_sigma ${{inputs.prior_sigma}} --prior_scale_ab ${{inputs.prior_scale_ab}} --squeeze_n ${{inputs.squeeze_n}} --n_segments ${{search_space.n_segments}}


## License
[MIT license](https://github.com/alherit/calibrated-betamix/blob/master/LICENSE).



