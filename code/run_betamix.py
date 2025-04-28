import os
import multiprocessing

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
    multiprocessing.cpu_count()
)

import jax

print("#devices: ", len(jax.devices()))

from jax import config
config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

import numpy as np
import jax.numpy as jnp
from jax import random, jit, nn, random, scipy, vmap


from tqdm import tqdm

from functools import partial


import os
import argparse

import mlflow

from scipy import stats as scipy_stats

import typing

from einops import rearrange, reduce, repeat

import util
import model
from util import *
from model import *


import sys
gettrace= sys.gettrace()

# For debugging
debug_status=True if gettrace else False
config.update("jax_debug_nans", True)
#config.update("jax_debug_nans", False)


if debug_status:
    print("I'm being debugged: no JIT")
JIT =  not debug_status 

from pip import _internal
_internal.main(['list'])





import matplotlib
# no type 3 fonts
#matplotlib.rcParams['pdf.fonttype'] = 42

from matplotlib import pyplot as plt
matplotlib.use('Agg')
#matplotlib.use('MacOSX')


### PARSE ARGS

parser = argparse.ArgumentParser()

# model
parser.add_argument('--n_components', default=10,
                    help='Number of components.', nargs='?', type=int)
parser.add_argument('--hidden_layers', default=2,
                    help='Number of hidden layers', nargs='?', type=int)
parser.add_argument('--nodes', default=64,
                    help='Number of nodes per hidden layer', nargs='?', type=int)
parser.add_argument('--activation', default='erf',
                    help='activation function', nargs='?', type=str)

parser.add_argument('--stax', default=False, action='store_true')
parser.add_argument("--stax_parameterization", default='ntk', nargs='?', type=str)


parser.add_argument('--nocopula', default=False, action='store_true')
parser.add_argument('--nobridgebetas', default=False, action='store_true')

parser.add_argument('--sorted_locations', default=False, action='store_true')


parser.add_argument('--gmdn', default=False, action='store_true')
parser.add_argument('--gaussmix_with_scaling', default=False, action='store_true')


# SG MCMC

parser.add_argument('--library', default="sgmcmcjax",
                    help='posterior approximation library',nargs='?', type=str)


parser.add_argument('--sampler', default="sgmcmc",
                    help='sampler', nargs='?', type=str)

parser.add_argument('--sg', default=False, action='store_true')

parser.add_argument('--noncompiled', default=False, action='store_true')


parser.add_argument('--maxnorm', default=np.inf,
                    help='max norm for gradient clipping', nargs='?', type=float)


parser.add_argument('--do_map', default=False, action='store_true')
parser.add_argument('--no_prior', default=False, action='store_true')

parser.add_argument('--freeze_hidden_after_map', default=False, action='store_true')

parser.add_argument('--map_epochs', default=10,
                    help='epochs for do_map', nargs='?', type=int)

parser.add_argument('--map_step_size', default=None,
                    help='map step size', nargs='?', type=float)

parser.add_argument('--map_batch_size', default=32,
                    help='batch size for do_map', nargs='?', type=int)


parser.add_argument('--map_prior_sigma', default=1.,
                    help='', nargs='?', type=float)


parser.add_argument('--vi_samples', default=10,
                    help='samples for kl estimation for vi', nargs='?', type=int)


parser.add_argument('--test_M', default=100,
                    help='Monte carlo repeats for testing', nargs='?', type=int)

parser.add_argument('--num_warmup', default=None,
                    help='number of warmup samples : if None, use default formula (bad idea)', nargs='?', type=int)


parser.add_argument('--batch_size', default=None,
                    help='Batch size for SGMCMC', nargs='?', type=int)


parser.add_argument('--step_size', default=None,
                    help='mcmc (first) step size', nargs='?', type=float)

parser.add_argument('--log_step_size', default=None,
                    help='mcmc (first) step size', nargs='?', type=float)


parser.add_argument('--last_step_size', default=None,
                    help='mcmc last step size', nargs='?', type=float)


parser.add_argument('--leap_frog_steps', default=None,
                    help='leap frog steps: if specified uses sghmc, otherwise sgldadam', nargs='?', type=int)

parser.add_argument('--num_chains', default=4,
                    help='chains', nargs='?', type=int)

parser.add_argument('--keep_every', default=None,
                    help='chains', nargs='?', type=int)




parser.add_argument('--prior_sigma', default=1.,
                    help='', nargs='?', type=float)

parser.add_argument('--log_prior_sigma', default=None,
                    help='', nargs='?', type=float)



parser.add_argument('--prior_alpha_weights', default=1.,
                    help='', nargs='?', type=float)


parser.add_argument('--prior_scale_ab', default=None,
                    help='', nargs='?', type=float)

parser.add_argument('--log_prior_scale_ab', default=None,
                    help='', nargs='?', type=float)


parser.add_argument('--prior_corr_concentration', default=1.,
                    help='', nargs='?', type=float)

parser.add_argument("--lkj", default='onion', nargs='?', type=str)

parser.add_argument("--logbetacdf_N", default=1000, nargs='?', type=int)

parser.add_argument('--nocholcorr', default=False, action='store_true')



parser.add_argument('--alpha_quant', default=None,
                    help='', nargs='?', type=float)



parser.add_argument("--seed", default=0, nargs='?', type=int)




# data
parser.add_argument("--dataset", default='boston', nargs='?', type=str)
parser.add_argument("--data_folder", default="../DropoutUncertaintyExps/UCI_Datasets/", nargs='?', type=str)

parser.add_argument("--split", default=0, nargs='?', type=int)

parser.add_argument('--validation', default=False, action='store_true', help='split train in 80/20 and use 20 as test set, ignore original test set')

parser.add_argument('--skip_train_eval', default=False, action='store_true')


parser.add_argument('--squeeze_n', default=1,
                    help='gaussian mixture components to squeeze the data', nargs='?', type=int)

parser.add_argument("--squeeze", default='gaussmix', nargs='?', type=str)


parser.add_argument("--n", default=None, nargs='?', help='size for synth dataset', type=str) 

# plots
parser.add_argument('--plot_final', default=False, action='store_true')

parser.add_argument('--plot_prior', default=False, action='store_true')
parser.add_argument('--plot_prior2D', default=False, action='store_true')
parser.add_argument('--plot_orig_prior', default=False, action='store_true')
parser.add_argument('--maxdens', default=None, help='', nargs='?', type=float)




ARGS = parser.parse_args()

if ARGS.log_prior_sigma is not None:
    ARGS.prior_sigma = 10. ** ARGS.log_prior_sigma

if ARGS.log_step_size is not None:
    ARGS.step_size = 10. ** ARGS.log_step_size 

if ARGS.log_prior_scale_ab is not None:
    ARGS.prior_scale_ab = 10. ** ARGS.log_prior_scale_ab


print(ARGS)

MAXDENS = ARGS.maxdens

np.random.seed(ARGS.seed)

# define the inital state
rng_key = random.PRNGKey(ARGS.seed+2)


mlflow.start_run()
mlflow.log_params(vars(ARGS))

NORMALIZE_X = True #ARGS.activation != "rbf"
data, y_range_dgp = load_data(ARGS.dataset, ARGS.split, ARGS.data_folder, ARGS.n, NORMALIZE_X)

def get_subset(A, size):
    return A[np.random.randint(A.shape[0], size=size)]


Y_DIM = data.Y_train.shape[1]

input_size = data.X_train.shape[1] 

if ARGS.validation:
    #sample sub
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data.X_train, data.Y_train, test_size=0.33, random_state=42)
    data.X_train = X_train
    data.Y_train = y_train
    data.X_test = X_test
    data.Y_test = y_test
    data.orig_X_train = None
    data.orig_X_test = None
    ARGS.plot_orig_prior = False


if Y_DIM == 1:
    ARGS.nocopula = True
    ARGS.nocholcorr = True

STAX = ARGS.stax
if not STAX and ARGS.hidden_layers>1 and not ARGS.gmdn:
    print("not implemented")
    exit(1)

######### for plotting
SUB_SIZE = 9
np.random.seed(0)
sub_indices  = np.random.choice(data.X_test.shape[0], size=np.minimum(SUB_SIZE,data.X_test.shape[0]), replace=False)

one_y_obs = jnp.min(data.Y_train, axis=0)

if ARGS.gmdn: # for gaussmix

    min_y_train = one_y_obs*0. - 3.
    max_y_train = one_y_obs*0. + 3.

else: # forbetas
    min_y_train = one_y_obs*0. 
    max_y_train = one_y_obs*0. + 1.
##################

def build_bayesian_net(rng_key, X_test, input_size, hidden_layers, suffix, prior_sigma, prior_scale_ab, X_test_orig=None):

    mean_y = jnp.mean(data.Y_train, axis=0)
    std_y = jnp.std(data.Y_train, axis=0)


    if ARGS.gmdn:

        predict, predict_with_probits, params_structure,cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate, loglikelihood, logDens, mix_preds_gaussmix, log_dens_uncorr, init_fn, init_fns, get_outputs_from_last_hidden =  build_gaussian_modelD(ARGS.gaussmix_with_scaling, data.Y_train, ARGS.squeeze_n, input_size, ARGS.activation, hidden_layers, ARGS.nodes, ARGS.n_components, prior_sigma, STAX, ARGS.stax_parameterization, mean_y, std_y)
        batch_mix = vmap(mix_preds_gaussmix,0)
        mix_preds = mix_preds_gaussmix
        
        plot2D_ml_flow = partial(util.plot2D_ml_flow, logDens=logDens, batch_mix=batch_mix)

        loglikelihood = partial(loglikelihood, logDens=logDens, from_preds=False)
        loglikelihood_corr_from_preds = partial(loglikelihood, logDens=logDens, from_preds=True)
        loglikelihood_uncorr_from_preds = partial(loglikelihood, logDens=log_dens_uncorr, from_preds=True)
        init_diffeo = None


    else:
        #init_fn is the global init function for the network, init_fns is a list containing the init fun for each layer

        predict, predict_with_probits, params_structure,cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate, loglikelihood, logDens, mix_preds_betamix, log_dens_uncorr, init_fn, init_fns, init_diffeo, get_outputs_from_last_hidden =  build_modelD(data.Y_train, ARGS.squeeze_n, input_size, ARGS.activation, hidden_layers, ARGS.nodes, ARGS.n_components, prior_sigma, ARGS.prior_alpha_weights, prior_scale_ab, ARGS.prior_corr_concentration, ARGS.alpha_quant, STAX, ARGS.stax_parameterization, not ARGS.nocopula, not ARGS.nobridgebetas, ARGS.sorted_locations, ARGS.lkj, ARGS.logbetacdf_N, not ARGS.nocholcorr, ARGS.squeeze, mean_y, std_y)
        batch_mix = vmap(mix_preds_betamix,0)
        mix_preds = mix_preds_betamix
        plot2D_ml_flow = partial(util.plot2D_ml_flow, logDens=logDens, batch_mix=batch_mix)

        loglikelihood = partial(loglikelihood, logDens=logDens, from_preds=False)
        loglikelihood_corr_from_preds = partial(loglikelihood, logDens=logDens, from_preds=True)
        loglikelihood_uncorr_from_preds = partial(loglikelihood, logDens=log_dens_uncorr, from_preds=True)

    logprior = model.logprior

    if JIT:
        predict = jit(predict)
        loglikelihood = jit(loglikelihood)
        loglikelihood_corr_from_preds = jit(loglikelihood_corr_from_preds)
        loglikelihood_uncorr_from_preds = jit(loglikelihood_uncorr_from_preds)

        mix_preds = jit(mix_preds)
        logprior = jit(logprior)


    if not STAX:
        print("NO STAX")
        print("number of parameters in the network: ", sum([s[0]+s[1] for s in jax.tree_util.tree_map(lambda n:np.prod(n.shape), params_structure)]))

        # parameters of the prior
        prior_params = jax.tree_util.tree_map(lambda coeff: jnp.stack([jnp.abs(coeff*0.),jnp.abs(coeff*0.+ prior_sigma)]), params_structure)
        logprior = partial(logprior, prior_params= prior_params)
        sample_prior_params = None

    else:
        print("USING STAX")

        prior_params = None

        def logprior(params, input_shape):
            logp = 0.
            for (init_fun,param) in zip(init_fns,params):
                if len(param) > 0:
                    input_shape, lp = init_fun(None, input_shape, param)
                    logp += lp
                else:#no params in this layer
                    input_shape, _ = init_fun(None, input_shape)
            return logp
        
        logprior = partial(logprior, input_shape=(-1, input_size))


        def sample_prior_params_conditional_net(key, M):
            v_initfn = vmap(init_fn,(0,None)) 
            sample_keys = jax.random.split(key,M)
            _, res = v_initfn(sample_keys,(-1, input_size))
            return res

        sample_prior_params = sample_prior_params_conditional_net


    if JIT:
        logprior = jit(logprior)


    ll = np.mean(logdens_gaussmix_multivariate(data.Y_test))
    print("unconditional scoring of test set ", ll)
    mlflow.log_metric("testset_uncond_loglik_corr", ll)




    batch_predict = vmap(predict, in_axes=(None, 0))  # over batch dimension of x


    ### Plot samples from prior before training

    batch_predict_with_probits = vmap(predict_with_probits, in_axes=(None, 0))  # over batch dimension of x
    double_batch_predict_with_probits = vmap(batch_predict_with_probits, in_axes=(0, None), out_axes=1) # over sampled parameters 


    rng_key, network_rng_key = jax.random.split(rng_key,2)
    if STAX:
        
        sampled_weights = sample_prior_params(network_rng_key, M_PLOT)
    else:
        sampled_weights = tree_random_normal(network_rng_key, prior_params, M_PLOT)

    X_test_subsample = X_test[sub_indices]


    preds, probits = double_batch_predict_with_probits(sampled_weights, X_test_subsample)

    batch_loglikelihood = vmap(loglikelihood, (None, 0, 0))
    
    plt.hist(probits.reshape(-1), bins=100, density=True)
    plot_save_mlflow('probits_'+suffix)

    if not ARGS.gmdn:
        plt.hist(preds.alphas.reshape(-1), bins=100, density=True)
        plot_save_mlflow('alphas_'+suffix)

        plt.hist(preds.betas.reshape(-1), bins=100, density=True)
        plot_save_mlflow('betas_'+suffix)

        plt.hist(jnp.exp(preds.logweights.reshape(-1)), bins=50, density=True)
        plot_save_mlflow('weights_'+suffix)

        if Y_DIM>1:
            corr = preds.corr @ jnp.transpose(preds.corr, (0,1,2,4,3))
            off_diag_corr= vmap(vmap(vmap(lambda mat: mat[jnp.where(~jnp.eye(corr.shape[3],dtype=bool))])))(corr)

            plt.hist(off_diag_corr.reshape(-1), bins=50)
            plot_save_mlflow('corr_'+suffix)

    
    Xcont = jnp.arange(-10.,10.,0.1)

    Xcont = jnp.stack([Xcont]*X_test_subsample.shape[1],axis=1)
    preds, probits = double_batch_predict_with_probits(sampled_weights, Xcont)

    for i in range(20):
        plt.plot(Xcont[:,0],probits[:,i,0])
    savefigs("gp_v")

    print("plotting prior")

    if ARGS.plot_prior:

        try:
            if True:
                if ARGS.gmdn: # for gaussmix

                    min_y_train = one_y_obs*0. - 3.
                    max_y_train = one_y_obs*0. + 3.

                else: # forbetas
                    min_y_train = one_y_obs*0. 
                    max_y_train = one_y_obs*0. + 1.


                for i in range(Y_DIM):
                    util.old_plot_ml_flow(preds, 'trained_prior{0}'.format(i) + suffix,  min_y_train[i], max_y_train[i], maxdens=MAXDENS, logDens=partial(log_dens_uncorr, covariate=i), batch_mix=batch_mix) # for 1D marginals
        
                if ARGS.plot_prior2D and Y_DIM==2: # this takes long
                    if True: #orig_scale
                        # this is just for the plot range
                        min_y_train = vmap(jnp.min,1)(data.Y_train)
                        max_y_train = vmap(jnp.max,1)(data.Y_train)
                        #X_test_orig_subsample = X_test_orig[sub_indices]
                        plot2D_ml_flow(preds, 'trained_prior2D' + suffix, min_y_train, max_y_train, x = None, cdf_squeeze = cdf_squeeze, logpdf_unsqueeze = logpdf_unsqueeze) 
                    else:
                        plot2D_ml_flow(preds, 'trained_prior2D' + suffix, min_y_train, max_y_train) 


            # plot prior predictives in original space
            # M sample for each x of the training set, average over x and w
            if ARGS.plot_orig_prior and Y_DIM==1: 
                rng_key, network_rng_key = jax.random.split(rng_key,2)
                M = M_PLOT
                if STAX:
                    sampled_weights = sample_prior_params(network_rng_key, M)
                else:
                    sampled_weights = tree_random_normal(network_rng_key, prior_params, M)


    #######################################################
                if ARGS.dataset in ["synth","dgp"]:
                    ### TODO: THIS CODE IS THE SAME AS THE ONE USED FOR THE POSTERIOR => REFACTOR 

                    if ARGS.gmdn:
                        def reorder(preds):
                            mus = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b nc m n d"), preds.mus)
                            covs = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b nc m n d"), preds.covs)
                            logweights = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n ->  b nc m n"), preds.logweights)
                            return GaussMix(logweights=logweights, mus=mus, covs=covs)
                        def rearr(preds):
                            mus = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b (nc m) n d"), preds.mus)
                            covs = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b (nc m) n d"), preds.covs)
                            logweights = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n ->  b (nc m) n"), preds.logweights)
                            return GaussMix(logweights=logweights, mus=mus, covs=covs)

                    else:
                        def reorder(preds):
                            alphas = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b nc m n d"), preds.alphas)
                            betas = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b nc m n d"), preds.betas)
                            logweights = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n ->  b nc m n"), preds.logweights)
                            corr = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d1 d2 ->  b nc m n d1 d2"), preds.corr)
                            return BetaMix(logweights=logweights, alphas=alphas, betas=betas, corr=corr)
                        def rearr(preds):
                            alphas = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b (nc m) n d"), preds.alphas)
                            betas = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b (nc m) n d"), preds.betas)
                            logweights = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n ->  b (nc m) n"), preds.logweights)
                            corr = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d1 d2 ->  b (nc m) n d1 d2"), preds.corr)
                            return BetaMix(logweights=logweights, alphas=alphas, betas=betas, corr=corr)



                    # I have to vmap inside over chain and M
                    vv_batch_predict = vmap(vmap(batch_predict,(0,None)),(0,None))

                    if JIT:
                        vv_batch_predict = jit(vv_batch_predict)


                    def chunkyfied_mix_preds(samples, X, y, loglik_from_preds, CHUNK_SIZE =  500):
                        
                        batch_loglik_from_preds = vmap(loglik_from_preds, in_axes=(None,0,0))
                        
                        all_ll = []
                        all_preds = []
                        for chunk_x,chunk_y in tqdm(spliterate((X,y), CHUNK_SIZE, y.shape[0])): # e.g. chunk_preds.alphas : (CHUNK_SIZE, 1, 400, 10, 2) (1 chain)
                            chunk_preds = vv_batch_predict(samples, chunk_x)
                            rearr_preds =  rearr(chunk_preds) # we get shape: b ncm n ..
                            mixed_preds = batch_mix(rearr_preds) # for each element of the batch, mix over ncm (number of chains x M)

                            ll = batch_loglik_from_preds(None, mixed_preds, chunk_y)
                            
                            all_preds.append(reorder(chunk_preds))            
                            all_ll.append(ll)
                            jax.clear_caches()
                        
                        return tree_concat(all_ll, axis = 0), all_preds  ## all_preds in list format. tree_concat consumes too much memory

                    sigma_str = "_sigma" + str(ARGS.prior_sigma)
                    scale_str = "_scale" + str(ARGS.prior_scale_ab)
                    params_str_image = "_gmdn"+sigma_str if ARGS.gmdn else sigma_str + scale_str
                    #expand_dims to add dummy chain dimension
                    plot_image(tree_map(lambda e: jnp.expand_dims(e,1), sampled_weights), [-4.,4.], [-4.,4.],  partial(chunkyfied_mix_preds, loglik_from_preds = loglikelihood_corr_from_preds), prefix="prior" + params_str_image ) # for dgp-like



    #######################################################
                SUB_SIZE_TRAIN = 200
                sub_indices_train  = np.random.choice(data.X_train.shape[0], size=np.minimum(SUB_SIZE_TRAIN,data.X_train.shape[0]), replace=False)
                sub_train_data = data.X_train[sub_indices_train]
                preds, probits = double_batch_predict_with_probits(sampled_weights, sub_train_data) # data.X_train) #[:50,:]

                # transform batch dimension B into sampling dimension M , i.e M <- B*M

                if ARGS.gmdn:
                    #B = preds.logweights.shape[0]
                    logweights = rearrange(preds.logweights, 'b c d-> (b c) d') #- jnp.log(B)
                    mus = rearrange(preds.mus, 'b c d e -> (b c) d e ')
                    covs = rearrange(preds.covs, 'b c d e -> (b c) d e ') 

                    remixed_preds = GaussMix(logweights=logweights,mus=mus,covs=covs) 

                else:
                    #B = preds.logweights.shape[0]
                    logweights = rearrange(preds.logweights, 'b c d -> (b c) d ') #- jnp.log(B)
                    alphas = rearrange(preds.alphas, 'b c d e-> (b c) d e')
                    betas = rearrange(preds.betas, 'b c d e-> (b c) d e') 
                    corr = rearrange(preds.corr, 'b c d1 d2 e -> (b c) d1 d2 e') 


                    remixed_preds = BetaMix(logweights=logweights,alphas=alphas,betas=betas, corr=corr) 


                # add missing dimension to make it appear as 1 observation
                #B=1
                remixed_preds = tree_map(partial(jnp.expand_dims, axis=0), remixed_preds)

                # make it smaller to avoid explosion
                #remixed_preds = tree_map(lambda e: e[:,:10000], remixed_preds)

                logpdf_uncond = logdens_gaussmix_multivariate #logpdf_unsqueeze
                sigma_str = "_sigma" + str(ARGS.prior_sigma)
                scale_str = "_scale" + str(ARGS.prior_scale_ab)
                suffix += "_gmdn"+sigma_str if ARGS.gmdn else sigma_str + scale_str  
                suffix += "_" + ARGS.dataset

                util.plot_ml_flow(remixed_preds,'origy_trained_prior' + suffix, logpdf_uncond=logpdf_uncond, logDens=logDens, batch_mix=batch_mix, minval = None, maxval = None, cdf_squeeze = cdf_squeeze, logpdf_unsqueeze = logpdf_unsqueeze, orig_ys=jnp.unique(jnp.squeeze(jnp.concatenate([data.Y_train, data.Y_test]))), r_factor=3)

        except:
            pass



    return rng_key, logprior, init_diffeo, sample_prior_params, prior_params, params_structure, batch_mix, batch_loglikelihood, batch_predict, predict, predict_with_probits, get_outputs_from_last_hidden, logDens, log_dens_uncorr, loglikelihood, loglikelihood_corr_from_preds, loglikelihood_uncorr_from_preds, cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate, plot2D_ml_flow

M_PLOT = 100



### MAP 
if ARGS.do_map: 

    rng_key, logprior, init_diffeo, sample_prior_params, prior_params, params_structure, batch_mix,  batch_loglikelihood, batch_predict, predict, predict_with_probits, get_outputs_from_last_hidden, logDens, log_dens_uncorr, loglikelihood, loglikelihood_corr_from_preds, loglikelihood_uncorr_from_preds, cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate, plot2D_ml_flow = build_bayesian_net(rng_key, data.X_test, input_size, ARGS.hidden_layers, "map", ARGS.map_prior_sigma, ARGS.prior_scale_ab, data.orig_X_test)


    from jax.example_libraries import optimizers
    opt_init, opt_update, get_params = optimizers.adam(ARGS.map_step_size)

    rng_key, network_rng_key = jax.random.split(rng_key,2)
    if STAX:
        initial_params = sample_prior_params(network_rng_key, 1)
    else:
        initial_params = tree_random_normal(network_rng_key, prior_params, 1)

    initial_params = tree_map(lambda e: jnp.squeeze(e,axis=0),initial_params)

    if ARGS.squeeze=="nn":
        diffeo_key, rng_key = jax.random.split(rng_key,2)
        diffeo_params = init_diffeo(diffeo_key, jnp.array([.5,.5]))
        initial_params = [diffeo_params,initial_params]

    opt_state = opt_init(initial_params)


    def loss_fn(params, X, y, full_train_size):
        if ARGS.no_prior: ##JUST ML
            return -jnp.mean(batch_loglikelihood(params, X, y), axis=0) 
        else:
            return -jnp.sum(batch_loglikelihood(params, X, y), axis=0) - (X.shape[0]/full_train_size) * logprior(params[1])
        
    loss_fn = partial(loss_fn, full_train_size=data.X_train.shape[0])

    if JIT:
        loss_fn = jit(loss_fn)


    def make_zero(grad):
        return jax.tree_util.tree_map(lambda e: e*0., grad)

    def step_fn(step, opt_state, x, y):
        value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state),x,y)

        if ARGS.squeeze=="nn":
            # alternate training
            if step%2 == 0:
                grads = [make_zero(grads[0]), grads[1]]
            else:
                grads = [grads[0], make_zero(grads[1])]
        
        opt_state = opt_update(step, grads, opt_state)

        return value, opt_state


    #loader
    import tensorflow as tf
    from tensorflow import data as tfdata
    import tensorflow_datasets  as tfds

    tf.random.set_seed(ARGS.seed+1) ## need this for dataset iteration using tf

    train_ds = tfdata.Dataset.from_tensor_slices((data.X_train,data.Y_train)).shuffle(10000).batch(ARGS.map_batch_size)

    # i got an error when using directly the same tensors with gpu, so let's convert them to numpy
    train_ds = tfds.as_numpy(train_ds) 


    step = 0
    for i in range(ARGS.map_epochs):
        loss = 0.
        j=0
        for x,y in tqdm(train_ds):

            loss_val, opt_state = step_fn(step, opt_state, x, y)

            loss += loss_val
            step += 1
            j += 1

        print("epoch: {:n} loss {:.3f}".format(i, loss/j))
        mlflow.log_metric("training_MAP_loss", loss/j, step=i)

        w_map = get_params(opt_state)


        ll_train = batch_loglikelihood(w_map, data.X_train, data.Y_train)
        print("MAP Train loglikelihood: ", ll_train.mean())
        mlflow.log_metric("trainset_loglik_map", ll_train.mean(),step=i)


        ll_test = batch_loglikelihood(w_map, data.X_test, data.Y_test)
        print("MAP Test loglikelihood: ", ll_test.mean())
        mlflow.log_metric("testset_loglik_map", ll_test.mean(),step=i)

    if ARGS.squeeze=="nn":
        w_map_diffeo = w_map[0]
        w_map = w_map[1]

        cdf_squeeze = partial(cdf_squeeze, params=w_map_diffeo) 
        logpdf_unsqueeze = partial(logpdf_unsqueeze, params=w_map_diffeo) 

    else:
        w_map_diffeo = None
    

  

    print("Test loglikelihood: ", ll_test.mean())
    mlflow.log_metric("testset_loglik", ll_test.mean())

    print("Train loglikelihood: ", ll_train.mean())
    mlflow.log_metric("trainset_loglik", ll_train.mean())

    X_test_subsample = data.X_test[sub_indices]

    plot_preds_test = batch_predict(w_map,X_test_subsample)

    plot_preds_test = tree_map(lambda e:jnp.expand_dims(e,1),plot_preds_test) ## add M dimension
    
    if Y_DIM==2:
        X_test_orig_subsample = data.orig_X_test[sub_indices]
        min_y_train = vmap(jnp.min,1)(data.Y_train)
        max_y_train = vmap(jnp.max,1)(data.Y_train)

        plot2D_ml_flow(plot_preds_test, 'final2D_MAP', min_y_train, max_y_train, x = X_test_orig_subsample, cdf_squeeze = cdf_squeeze, logpdf_unsqueeze = logpdf_unsqueeze) 

    if ARGS.dataset != "dgp" and Y_DIM==1:

        # set this to make plots on the original scale
        minval = jnp.min(data.Y_test)
        maxval = jnp.max(data.Y_test)
        plot_ml_flow = partial(util.plot_ml_flow, minval = minval, maxval = maxval, cdf_squeeze = cdf_squeeze, logpdf_unsqueeze = logpdf_unsqueeze, orig_ys=jnp.unique(jnp.squeeze(jnp.concatenate([data.Y_train, data.Y_test]))), r_factor=3)
        plot_ml_flow_x = partial(util.plot_ml_flow_x, minval = minval, maxval = maxval, cdf_squeeze = cdf_squeeze, logpdf_unsqueeze = logpdf_unsqueeze, orig_ys=jnp.unique(jnp.squeeze(jnp.concatenate([data.Y_train, data.Y_test]))), r_factor=3)

       
        ### need to fix coord stuff for univariate
        plot_ml_flow(plot_preds_test,'final_MAP', logDens=logDens, batch_mix=batch_mix)

        if ARGS.dataset == "synth":
            plot_ml_flow_x(plot_preds_test,'final_MAP_x', data.X_test[sub_indices], logDens=logDens, batch_mix=batch_mix)


    if ARGS.freeze_hidden_after_map and ARGS.num_warmup>0: ## continue with MCMC AFTER MAP with FROZEN LAYERS  

        # compute new features
        data.X_train = get_outputs_from_last_hidden(w_map,data.X_train)
        data.X_test = get_outputs_from_last_hidden(w_map,data.X_test)

        rng_key, logprior, init_diffeo, sample_prior_params, prior_params, params_structure, batch_mix,  batch_loglikelihood, batch_predict, predict, predict_with_probits, get_outputs_from_last_hidden, logDens, log_dens_uncorr, loglikelihood, loglikelihood_corr_from_preds, loglikelihood_uncorr_from_preds, cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate, plot2D_ml_flow = build_bayesian_net(rng_key, data.X_test, ARGS.hidden_layers, ARGS.hidden_layers, "partial", ARGS.prior_sigma, ARGS.prior_scale_ab)


    if ARGS.squeeze=="nn":
        loglikelihood_w_diffeo = partial(loglikelihood, logDens=logDens, from_preds=False, params_diffeo=w_map_diffeo)
        loglikelihood_corr_from_preds = partial(loglikelihood, logDens=logDens, from_preds=True, params_diffeo=w_map_diffeo)
        loglikelihood_uncorr_from_preds = partial(loglikelihood, logDens=log_dens_uncorr, from_preds=True, params_diffeo=w_map_diffeo)
        loglikelihood = loglikelihood_w_diffeo



if ARGS.num_warmup>0:

    if not (ARGS.do_map and ARGS.freeze_hidden_after_map):
        rng_key, logprior, init_diffeo, sample_prior_params, prior_params, params_structure, batch_mix,  batch_loglikelihood, batch_predict, predict, predict_with_probits, get_outputs_from_last_hidden, logDens, log_dens_uncorr, loglikelihood, loglikelihood_corr_from_preds, loglikelihood_uncorr_from_preds, cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate, plot2D_ml_flow = build_bayesian_net(rng_key, data.X_test, input_size, ARGS.hidden_layers, "full_stochastic", ARGS.prior_sigma, ARGS.prior_scale_ab)


    total = 0
    for layer in params_structure:
        for param in layer:
            total += param.size 
    print("TOTAL NUMBER OF PARAMS:", total)

    ### POSTERIOR APPROXIMATION
         


    if ARGS.library == "blackjax":
        samples = sample_posterior_blackjax(ARGS.validation, rng_key, loglikelihood, logprior, ARGS.num_chains, ARGS.num_warmup, ARGS.test_M, ARGS.keep_every, ARGS.sg, ARGS.sampler, ARGS.step_size, ARGS.vi_samples, ARGS.batch_size, ARGS.leap_frog_steps, data.X_train, data.Y_train, prior_params, sample_prior_params, params_structure, ARGS.maxnorm)
    elif ARGS.library == "sgmcmcjax":
        samples = sample_posterior_sgmcmcjax(ARGS.validation, rng_key, loglikelihood, logprior, ARGS.num_chains, ARGS.num_warmup, ARGS.test_M, ARGS.keep_every, ARGS.step_size, ARGS.batch_size, ARGS.leap_frog_steps, data.X_train, data.Y_train, prior_params, sample_prior_params, ARGS.maxnorm, ARGS.noncompiled)
    else: #not implemented
        exit()


    if True:

        if ARGS.gmdn:
            def reorder(preds):
                mus = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b nc m n d"), preds.mus)
                covs = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b nc m n d"), preds.covs)
                logweights = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n ->  b nc m n"), preds.logweights)
                return GaussMix(logweights=logweights, mus=mus, covs=covs)
            def rearr(preds):
                mus = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b (nc m) n d"), preds.mus)
                covs = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b (nc m) n d"), preds.covs)
                logweights = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n ->  b (nc m) n"), preds.logweights)
                return GaussMix(logweights=logweights, mus=mus, covs=covs)

        else:
            def reorder(preds):
                alphas = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b nc m n d"), preds.alphas)
                betas = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b nc m n d"), preds.betas)
                logweights = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n ->  b nc m n"), preds.logweights)
                corr = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d1 d2 ->  b nc m n d1 d2"), preds.corr)
                return BetaMix(logweights=logweights, alphas=alphas, betas=betas, corr=corr)
            def rearr(preds):
                alphas = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b (nc m) n d"), preds.alphas)
                betas = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d ->  b (nc m) n d"), preds.betas)
                logweights = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n ->  b (nc m) n"), preds.logweights)
                corr = jax.tree_util.tree_map(lambda x: rearrange (x, "m nc b n d1 d2 ->  b (nc m) n d1 d2"), preds.corr)
                return BetaMix(logweights=logweights, alphas=alphas, betas=betas, corr=corr)



        # I have to vmap inside over chain and M
        vv_batch_predict = vmap(vmap(batch_predict,(0,None)),(0,None))

        if JIT:
            vv_batch_predict = jit(vv_batch_predict)


        def chunkyfied_mix_preds(samples, X, y, loglik_from_preds, CHUNK_SIZE =  500):
            
            batch_loglik_from_preds = vmap(loglik_from_preds, in_axes=(None,0,0))
            
            all_ll = []
            all_preds = []
            for chunk_x,chunk_y in tqdm(spliterate((X,y), CHUNK_SIZE, y.shape[0])): # e.g. chunk_preds.alphas : (CHUNK_SIZE, 1, 400, 10, 2) (1 chain)
                chunk_preds = vv_batch_predict(samples, chunk_x)
                rearr_preds =  rearr(chunk_preds) # we get shape: b ncm n ..
                mixed_preds = batch_mix(rearr_preds) # for each element of the batch, mix over ncm (number of chains x M)

                ll = batch_loglik_from_preds(None, mixed_preds, chunk_y)
                
                all_preds.append(reorder(chunk_preds))            
                all_ll.append(ll)
                jax.clear_caches()
            
            return tree_concat(all_ll, axis = 0), all_preds  ## all_preds in list format. tree_concat consumes too much memory



        print("computing loglik...")


        CHUNK_SIZE = 10
        #ll_test2  = ll_mixed_preds(preds_test, data.Y_test)
        ll_test, preds_test   = chunkyfied_mix_preds(samples, data.X_test, data.Y_test, loglik_from_preds = loglikelihood_corr_from_preds , CHUNK_SIZE =  CHUNK_SIZE)
        print("Test loglikelihood: ", ll_test.mean())
        mlflow.log_metric("testset_loglik", ll_test.mean())


        if Y_DIM>1 and not ARGS.validation:
            print("mixing chains test uncorr...")
            ll_test, _   = chunkyfied_mix_preds(samples, data.X_test, data.Y_test, loglik_from_preds = loglikelihood_uncorr_from_preds ,CHUNK_SIZE =  CHUNK_SIZE)
            print("Test loglikelihood (uncorrelated): ", ll_test.mean())
            mlflow.log_metric("testset_loglik_uncorr", ll_test.mean())

        if not ARGS.validation and not ARGS.skip_train_eval: # quick fix for hyperparameter tuning, this consumes a lot of memory and fails sometimes
            print("mixing chains train corr...")
            #ll_train2 = ll_mixed_preds(preds_train, data.Y_train)
            ll_train, _ = chunkyfied_mix_preds(samples, data.X_train, data.Y_train, loglik_from_preds = loglikelihood_corr_from_preds, CHUNK_SIZE =  CHUNK_SIZE)
            print("Train loglikelihood: ", ll_train.mean())
            mlflow.log_metric("trainset_loglik", ll_train.mean())



    preds_test = tree_concat(preds_test, axis = 0) # this consumes too much memory if dataset is big

    if ARGS.squeeze == "nn":
        cdf_squeeze = partial(cdf_squeeze, params=w_map_diffeo) 
        logpdf_unsqueeze = partial(logpdf_unsqueeze, params=w_map_diffeo) 



    if ARGS.plot_final:
        print("plotting final")
        if ARGS.sampler == "vi": 
            plot_preds_test = tree_map(lambda e: e[sub_indices,:,:(M_PLOT)], preds_test)
        else:
            plot_preds_test = tree_map(lambda e: e[sub_indices,:,:(M_PLOT//ARGS.num_chains)], preds_test)


        if ARGS.gmdn:
            def group(preds):
                mus = jax.tree_util.tree_map(lambda x: rearrange (x, "b nc m n d ->  b (nc m) n d"), preds.mus)
                covs = jax.tree_util.tree_map(lambda x: rearrange (x, "b nc m n d ->  b (nc m) n d"), preds.covs)
                logweights = jax.tree_util.tree_map(lambda x: rearrange (x, "b nc m n ->  b (nc m) n"), preds.logweights)
                return GaussMix(logweights=logweights, mus=mus, covs=covs)

        else:
            def group(preds):
                alphas = jax.tree_util.tree_map(lambda x: rearrange (x, "b nc m n d ->  b (nc m) n d"), preds.alphas)
                betas = jax.tree_util.tree_map(lambda x: rearrange (x, "b nc m n d ->  b (nc m) n d"), preds.betas)
                logweights = jax.tree_util.tree_map(lambda x: rearrange (x, "b nc m n ->  b (nc m) n"), preds.logweights)
                corr = jax.tree_util.tree_map(lambda x: rearrange (x, "b nc m n d1 d2 ->  b (nc m) n d1 d2"), preds.corr)
                return BetaMix(logweights=logweights, alphas=alphas, betas=betas, corr=corr)

        plot_preds_test = group(plot_preds_test)


        if Y_DIM==1: #ARGS.dataset != "dgp" and Y_DIM==1:

            # set this to make plots on the original scale
            minval = jnp.min(data.Y_test)
            maxval = jnp.max(data.Y_test)
            plot_ml_flow = partial(util.plot_ml_flow, minval = minval, maxval = maxval, cdf_squeeze = cdf_squeeze, logpdf_unsqueeze = logpdf_unsqueeze, orig_ys=jnp.unique(jnp.squeeze(jnp.concatenate([data.Y_train, data.Y_test]))), r_factor=3)
            plot_ml_flow_x = partial(util.plot_ml_flow_x, minval = minval, maxval = maxval, cdf_squeeze = cdf_squeeze, logpdf_unsqueeze = logpdf_unsqueeze, orig_ys=jnp.unique(jnp.squeeze(jnp.concatenate([data.Y_train, data.Y_test]))), r_factor=3)

            plot_ml_flow(plot_preds_test,'final', logDens=logDens, batch_mix=batch_mix)


            if ARGS.dataset == "synth" or ARGS.dataset == "dgp":
                plot_ml_flow_x(plot_preds_test,'final_x', data.X_test[sub_indices], logDens=logDens, batch_mix=batch_mix)
                sigma_str = "_sigma" + str(ARGS.prior_sigma)
                scale_str = "_scale" + str(ARGS.prior_scale_ab)
                params_str_image = "_gmdn"+sigma_str if ARGS.gmdn else sigma_str + scale_str

                plot_image(samples, [-4.,4.], [-4.,4.],  partial(chunkyfied_mix_preds, loglik_from_preds = loglikelihood_corr_from_preds), prefix="posterior" + params_str_image) # for dgp-like image

        for i in range(Y_DIM):
            print("marginal: ",i)
            old_plot_ml_flow(plot_preds_test, 'final{0}'.format(i), min_y_train[i], max_y_train[i], maxdens=MAXDENS, logDens=partial(log_dens_uncorr, covariate=i), batch_mix=batch_mix) # for 1D marginals

        if Y_DIM==2:
            X_test_orig_subsample = data.orig_X_test[sub_indices]
            min_y_train = vmap(jnp.min,1)(data.Y_train)
            max_y_train = vmap(jnp.max,1)(data.Y_train)

            plot2D_ml_flow(plot_preds_test, 'final2D', min_y_train, max_y_train, x = X_test_orig_subsample, cdf_squeeze = cdf_squeeze, logpdf_unsqueeze = logpdf_unsqueeze) 


    Y_test_squeezed = vmap(cdf_squeeze)(data.Y_test)

    if ARGS.sampler != "vi" and not ARGS.validation:
        # compute r_hat on test set
        import arviz
        print("computing r_hat...")
        if True: # shape: b nc m n (d1) (d2)  
            batch_logDens = vmap(vmap(vmap(logDens, (0,0)),(None,1)),(None,1)) # compute logDens for each mixture and y of the dataset

            if JIT:
                batch_logDens = jit(batch_logDens)



            def chunkyfied_batchLogDens(squeezed_y, preds, CHUNK_SIZE =  50):
                all_ll = []
                for chunk_y,chunk_preds in tqdm(spliterate((squeezed_y,preds), CHUNK_SIZE, squeezed_y.shape[0])):
                    ll = batch_logDens(chunk_y,chunk_preds) # e.g. chunk_preds.alphas : (CHUNK_SIZE, 1, 400, 10, 2) (1 chain)
                    all_ll.append(ll)
                
                return tree_concat(all_ll, axis = 2)


            ll = chunkyfied_batchLogDens(Y_test_squeezed, preds_test , CHUNK_SIZE=CHUNK_SIZE)

        ll = arviz.convert_to_dataset(np.array(ll))

        r_hat = arviz.rhat(ll)

        r_hat = np.array(r_hat["x"])


        r_hat_mean = float(r_hat.mean())
        r_hat_std = float(r_hat.std())
        print("R-hat: mean {:.4f} std {:.4f}".format(r_hat_mean, r_hat_std))
        mlflow.log_metric("r_hat_mean", r_hat_mean)
        mlflow.log_metric("r_hat_std", r_hat_std)


if not ARGS.validation:
    v_cdf_squeeze = vmap(cdf_squeeze)
    Y_train_squeezed = v_cdf_squeeze(data.Y_train)
    Y_test_squeezed = v_cdf_squeeze(data.Y_test)

    print(jnp.min(Y_train_squeezed))
    print(jnp.max(Y_train_squeezed))

    if Y_DIM==2:
        plt.hist2d(data.Y_train[:,0],data.Y_train[:,1], bins=100)
        plot_save_mlflow('Y_train_original_2D')
        plt.hist2d(Y_train_squeezed[:,0],Y_train_squeezed[:,1], bins=100)
        plot_save_mlflow('Y_train_squeezed_2D')
        plt.hist2d(data.Y_test[:,0],data.Y_test[:,1], bins=100)
        plot_save_mlflow('Y_test_original_2D')
        plt.hist2d(Y_test_squeezed[:,0],Y_test_squeezed[:,1], bins=100)
        plot_save_mlflow('Y_test_squeezed_2D')


    if Y_DIM==1:
        Y_train_squeezed = jnp.expand_dims(Y_train_squeezed,1)
        Y_test_squeezed = jnp.expand_dims(Y_test_squeezed,1)

    for i in range(Y_DIM):


        plt.hist(jnp.squeeze(Y_train_squeezed[:,[i]]), bins=50)
        plot_save_mlflow('Y_train_squeezed{0}'.format(i))

        plt.hist(jnp.squeeze(Y_test_squeezed[:,[i]]), bins=50)
        plot_save_mlflow('Y_test_squeezed{0}'.format(i))



mlflow.end_run()


