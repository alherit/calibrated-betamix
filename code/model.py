import jax
from jax import numpy as jnp
from jax import vmap
import typing
from einops import rearrange, reduce, repeat


class BetaMix(typing.NamedTuple):
    logweights: jnp.ndarray
    alphas: jnp.ndarray
    betas: jnp.ndarray
    corr: jnp.ndarray




from sklearn.mixture import GaussianMixture
from functools import partial
import numpy as np
import matplotlib
# no type 3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42
from matplotlib import pyplot as plt
import mlflow
import os 
from scipy import stats as scipy_stats

from util import savefigs

from mydense import MyDense



#from einops import einsum

def train_gaussmix_squeeze(Y_train, n_components):

    GRID_SEARCH = False
    if GRID_SEARCH:
        from sklearn.model_selection import GridSearchCV

        def gmm_bic_score(estimator, X):
            """Callable to pass to GridSearchCV that will use the BIC score."""
            # Make it negative since GridSearchCV expects a score to maximize
            return -estimator.bic(X)


        param_grid = {
            "n_components": range(1, n_components),
            "covariance_type": ["spherical", "tied", "diag", "full"],
        }
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
        )
        grid_search.fit(Y_train)

        import pandas as pd

        df = pd.DataFrame(grid_search.cv_results_)[
            ["param_n_components", "param_covariance_type", "mean_test_score"]
        ]
        df["mean_test_score"] = -df["mean_test_score"]
        df = df.rename(
            columns={
                "param_n_components": "Number of components",
                "param_covariance_type": "Type of covariance",
                "mean_test_score": "BIC score",
            }
        )
        print(df.sort_values(by="BIC score").head())

        gm = grid_search.best_estimator_
    else:
        gm = GaussianMixture(n_components=n_components, random_state=0, init_params='k-means++', n_init=10, max_iter=10000).fit(Y_train)

    # coord is ignored
    def logpdf_unsqueeze(y,  mus, covs, logweights):
        assert len(y.shape)==1 

        logpdf_weighted_comp = logweights + jax.scipy.stats.multivariate_normal.logpdf(y,mus,covs)

        return jax.scipy.special.logsumexp(logpdf_weighted_comp,0)
        #return gm.score_samples(y) #equivalent but doesn't work on traced arrays


    # conditional CDF squeeze: y -> y'. coord is ignored
    def ccdf_squeeze(y, mus, covs, logweights):
        assert len(y.shape)==1

        squeezed_y = []

        for i in range(y.shape[0]):
            # a={i} , b={j:j<i}

            if i==0: ## no conditioning for first coordinate
                sigmas = jnp.sqrt(covs[:,i,i])
                log_cdf_weighted_comp = logweights + vmap(jax.scipy.stats.norm.logcdf,(None,0,0))(y[i], mus[:,i], sigmas)
                log_ccdf = jax.scipy.special.logsumexp(log_cdf_weighted_comp,0)

            else:
                x_a = y[i:(i+1)]
                x_b = y[:i]
                mus_a = mus[:,i:(i+1)]
                mus_b = mus[:,:i]
                covs_aa = covs[:,i:(i+1),i:(i+1)]  
                covs_ab = covs[:,i:(i+1),:i]  
                covs_bb = covs[:,:i,:i]
                covs_ba = covs[:,:i,i:(i+1)]

                covs_bb_inv = jnp.linalg.inv(covs_bb)

                # c for component
                covs_ab_covs_bb_inv = jnp.einsum("c a b, c b b -> c a b", covs_ab, covs_bb_inv)
                mus_a_given_b = mus_a + jnp.einsum("c a b, c b -> c a", covs_ab_covs_bb_inv, (x_b - mus_b))
                #covs_a_given_b = covs_aa - jnp.matmul(jnp.matmul(covs_ab, covs_bb_inv), covs_ba)
                covs_a_given_b = covs_aa - jnp.einsum("c a b, c b d -> c a d",  covs_ab_covs_bb_inv, covs_ba) # d=a , but i need to use a different letter for einsum

                log_pdf_xb_weighted_comp = logweights + jax.scipy.stats.multivariate_normal.logpdf(x_b,mus_b,covs_bb)
                #log_pdf_xb_weighted_comp2 = log_weights + vmap(jax.scipy.stats.multivariate_normal.logpdf, (None,0,0)) (x_b,mus_b,covs_bb)

                log_pdf_xb = jax.scipy.special.logsumexp(log_pdf_xb_weighted_comp,0)

                conditional_logweights = log_pdf_xb_weighted_comp - log_pdf_xb

                # sanity check: these weights should add up to 1:
                #jnp.sum(jnp.exp(conditional_logweights), axis=0)

                # log conditional cdf 
                log_ccdf_weighted_comp = conditional_logweights + vmap(jax.scipy.stats.norm.logcdf,(None,0,0)) (jnp.squeeze(x_a), jnp.squeeze(mus_a_given_b, axis=1), jnp.sqrt(jnp.squeeze(covs_a_given_b, axis=(1,2))))  #squeeze because a is 1-dimensional

                log_ccdf = jax.scipy.special.logsumexp(log_ccdf_weighted_comp,0)


            squeezed_y.append(jnp.exp(log_ccdf))

        squeezed_y = jnp.stack(squeezed_y)
        
        return squeezed_y

    log_weights = jnp.log(jnp.array(gm.weights_))
    ccdf_squeeze = partial(ccdf_squeeze, mus=jnp.array(gm.means_), covs=jnp.array(gm.covariances_), logweights=log_weights)
    logpdf_unsqueeze = partial(logpdf_unsqueeze, mus=jnp.array(gm.means_), covs=jnp.array(gm.covariances_), logweights=log_weights)


    return ccdf_squeeze, logpdf_unsqueeze, gm.score_samples


def train_gaussmix_squeeze_old(Y_train, n_components):
    gm = GaussianMixture(n_components=n_components, random_state=0, n_init=10, max_iter=10000).fit(Y_train)

    def logdens_gaussmix_1(y, mus, covs, logweights): #, coord=0):
        assert len(y.shape)==1 ##one column y element
        # get marginal distribution for coord
        #mus = mus[:,coord] 
        #sigmas = jnp.sqrt(covs[:,coord, coord])
        sigmas = jnp.sqrt(covs)

        logdens = jax.scipy.stats.norm.logpdf(y,mus,sigmas)
        logweightdens = logdens + logweights

        ll = jax.scipy.special.logsumexp(logweightdens)

        return ll


    def cdf_gaussmix_1(y, mus, covs, weights):#, coord=0):
        assert len(y.shape)==2 
        # get marginal distribution for coord
        #mus = mus[:,coord] 
        #sigmas = jnp.sqrt(covs[:,coord, coord])
        sigmas = jnp.sqrt(covs)

        cdf = jax.scipy.stats.norm.cdf(y,mus,sigmas)

        if len(weights.shape)>0:
            return jnp.expand_dims(cdf @ weights, axis=1) # return column vector
        else:
            return cdf

    def logdens_gaussmix(y, coord, mus, covs, logweights):
        if coord is None: #unsqueeze all coordinates
            covs = vmap(jnp.diag,0)(covs)
            logdens = jnp.squeeze(vmap(logdens_gaussmix_1,(0,1,1,None))(jnp.expand_dims(y,1), mus, covs, logweights))
            return jnp.sum(logdens)
        else:
            return logdens_gaussmix_1(y, mus[:,coord], covs[:,coord,coord], logweights)

    def cdf_gaussmix(y, coord, mus, covs, weights): 
        if coord is None: #squeeze all coordinates
            #for projections I need only the diagonal of the cov
            covs = vmap(jnp.diag,0)(covs)
            #return jnp.squeeze(vmap(cdf_gaussmix_1,(1,1,1,None), out_axes=1)(jnp.expand_dims(y,2), mus, covs, weights))
            return vmap(cdf_gaussmix_1,(1,1,1,None), out_axes=1)(jnp.expand_dims(y,2), mus, covs, weights) #vmap over covariates
        else:
            return cdf_gaussmix_1(y, mus[:,coord], covs[:,coord,coord], weights)



    # def cdf_gaussmix1(y, mus, sigmas, weights):
    #     assert len(y.shape)==1 
    #     cdf = jax.scipy.stats.norm.cdf(y,mus,sigmas)

    #     if len(weights.shape)>0:
    #         return jnp.expand_dims(cdf @ weights, axis=0) # return column vector
    #     else:
    #         return cdf


    logpdf_unsqueeze = partial(logdens_gaussmix, mus=gm.means_, covs=gm.covariances_, logweights=jnp.log(gm.weights_))  ## needs to be applied to the original y , not to the squeezed_y !!!
    #equivalent to: gm.score_samples



    cdf_gaussmix = partial(cdf_gaussmix, mus=gm.means_, covs=gm.covariances_, weights=gm.weights_)
    #cdf_gaussmix1 = partial(cdf_gaussmix1, mus=mus, sigmas=sigmas, weights=weights)

    for j in range(Y_train.shape[1]): #debug
        y_tr = Y_train[:,j]
        points_plus_one = 200
        y_max = jnp.max(y_tr)
        y_min = jnp.min(y_tr)
        step = (y_max-y_min)/points_plus_one

        ys = np.arange(y_min+step/2, y_max-step/2, step)

        ps = jnp.exp(vmap(logpdf_unsqueeze, (0,None))(np.expand_dims(ys,1), j))
        plt.plot(ys,ps, color='r')
        
        dist_name = "norm"
        dist = eval("scipy_stats." + dist_name)
        jax_dist = eval("jax.scipy.stats." + dist_name)
        loc, scale = dist.fit(y_tr)
        logpdf_unsqueeze1 = partial(jax_dist.logpdf,loc=loc,scale=scale)
        ps1 = jnp.exp(vmap(logpdf_unsqueeze1)(ys))
        plt.plot(ys,ps1, color='b')
        ##plt.show()



        cs = cdf_gaussmix(jnp.expand_dims(ys, 1), coord=j)
        #cs = vmap(cdf_gaussmix1,0)(jnp.expand_dims(ys, 1))
        plt.plot(ys,cs, color='r')
        
        cdf1 = partial(jax_dist.cdf,loc=loc,scale=scale)
        cs1 = vmap(cdf1)(ys)
        plt.plot(ys,cs1, color='b')
        #plt.show()

        plt.hist(jnp.squeeze(y_tr), bins=100, density=True)

        fname= "squeezing%i" %j
        savefigs(fname) #, bbox_inches='tight')
        #plt.close()
        #mlflow.log_artifact(fname)
        #   os.remove(fname) #don't leave garbage

    return cdf_gaussmix, logpdf_unsqueeze, gm.score_samples


from jax import random

def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


# sample normal variables on each node of the tree
def tree_random_normal(rng_key, target, batch_size):
    keys_tree = random_split_like_tree(rng_key, target)
    # return jax.tree_multimap(
    return jax.tree_util.tree_map(
        lambda mu_sigma, k: jax.random.normal(k,(batch_size,) + mu_sigma[0].shape) * mu_sigma[1] + mu_sigma[0],
        target,
        keys_tree,
    )

def build_layer( m, n):
    return ( jnp.zeros((n, m))),  jnp.zeros((n,))


def get_param_structure(sizes):
    return [build_layer( m, n) for  m, n in zip(sizes[:-1], sizes[1:])]


def expon_inverse_cdf(x):
    x = jnp.minimum(x, .999999999)
    return -jnp.log(1.-x)

def soft_round(x, alpha, eps=1e-3):
    """Differentiable approximation to `round`.

    Larger alphas correspond to closer approximations of the round function.
    If alpha is close to zero, this function reduces to the identity.

    This is described in Sec. 4.1. in the paper
    > "Universally Quantized Neural Compression"<br />
    > Eirikur Agustsson & Lucas Theis<br />
    > https://arxiv.org/abs/2006.09952

    Args:
    x: `tf.Tensor`. Inputs to the rounding function.
    alpha: Float or `tf.Tensor`. Controls smoothness of the approximation.
    eps: Float. Threshold below which `soft_round` will return identity.

    Returns:
    `tf.Tensor`
    """

    if alpha is None or alpha==0.:
        return x

    # This guards the gradient of tf.where below against NaNs, while maintaining
    # correctness, as for alpha < eps the result is ignored.
    alpha_bounded = jnp.maximum(alpha, eps)


    m = jnp.floor(x) + .5
    r = x - m
    z = jnp.tanh(alpha_bounded / 2.) * 2.
    y = m + jnp.tanh(alpha_bounded * r) / z


    # For very low alphas, soft_round behaves like identity
    return jnp.where(alpha < eps, x, y)

from sympy import Integer, Matrix






#STAX = True
def build_pred_logits(activation, prior_sigma, nodes, hidden_layers, output_size, input_size, STAX, parameterization, with_scaling=True):

    if STAX:
        from jax import random
        from neural_tangents import stax
        import neural_tangents

        activ = activation[0].upper() + activation[1:]
        
        # this matches perfectly my formula when taking kernel_fn for taxi dataset (6 inputs)
        #layers = [MyDense(nodes, W_std= prior_sigma * np.sqrt(6) , b_std= prior_sigma , parameterization='standard'),eval("stax." + activ)() ]*layers + [ MyDense(1, W_std= prior_sigma * np.sqrt(nodes), b_std= prior_sigma  , parameterization='standard')]
        
        
        if parameterization == 'standard_custom':
            parameterization = 'standard'
            layers = [MyDense(nodes, W_std= prior_sigma*jnp.sqrt(input_size)  , b_std= prior_sigma , parameterization=parameterization),eval("stax." + activ)() ]
            layers += [MyDense(nodes, W_std= prior_sigma*jnp.sqrt(nodes)  , b_std= prior_sigma , parameterization=parameterization),eval("stax." + activ)() ]*(hidden_layers-1) 
            layers += [ MyDense(output_size, W_std= prior_sigma*jnp.sqrt(nodes), b_std= prior_sigma  , parameterization=parameterization)]
        else:
            layers = [MyDense(nodes, W_std= prior_sigma  , b_std= prior_sigma , parameterization=parameterization),eval("stax." + activ)() ]*hidden_layers + [ MyDense(output_size, W_std= prior_sigma, b_std= prior_sigma  , parameterization=parameterization)]

        init_fn, apply_fn, kernel_fn = stax.serial(*layers)

        init_fns, apply_fns, kernel_fns = zip(*layers)


        _, get_output_from_last_hidden, _ = stax.serial(*(layers[:-1]))  
        #get_output_from_last_hidden = None

    else:
        if activation=="erf":
            activ_func = jax.lax.erf
        else:
            activ_func = eval("jax.nn." + activation)

        def neural_net(params, x, activ_func):
            # per-example predictions
            activations = x
            for w, b in params[:-1]:
                outputs = jnp.dot(w, activations) + b
                activations = activ_func(outputs)

            final_w, final_b = params[-1]
            logits = jnp.dot(final_w, activations) + final_b
            return logits

        apply_fn = partial(neural_net,  activ_func=activ_func)
        init_fn = None
        init_fns = None
        kernel_fn = None
        get_output_from_last_hidden = None

    def pred_normal_logits(params,x, kernel_fn, apply_fn):
        if STAX:
            params = jax.tree_util.tree_map(jnp.squeeze,params)
            x = jnp.expand_dims(x,0)

        logits_before = apply_fn(params,x)

        if STAX:
            logits_before = jnp.squeeze(logits_before)

        if STAX:

            #xx = jnp.expand_dims(x,0)
            #kernel_fn = neural_tangents.empirical_kernel_fn(pred_logits )
            #kernel = kernel_fn(xx, None, 'nngp', params)
            kernel = kernel_fn(x, x, 'nngp')
            scale_true = jnp.sqrt(jnp.diag(kernel))


        else:
            if activation == "relu":
                # this is correct, for one hidden layer 
                var_hidden = prior_sigma**2 * (1. + jnp.linalg.norm(x)**2) / 2.



            elif activation == "erf":
                x_tilde = jnp.concatenate([x, jnp.ones(1)])
                two_sq_norm = 2. * jnp.linalg.norm(x_tilde)**2
                var_hidden = (2./jnp.pi) * jnp.arcsin (two_sq_norm / (1. + two_sq_norm))

            else: # not implemented
                exit()

            var_true = prior_sigma**2 * (1. + nodes * var_hidden)
            scale_true = jnp.sqrt(var_true)



        if with_scaling:
            logits = (logits_before - 0.)/scale_true
        else:
            logits = logits_before
        return logits

    return partial(pred_normal_logits, kernel_fn=kernel_fn, apply_fn=apply_fn), init_fn, init_fns, get_output_from_last_hidden

import scipy.linalg
def get_L_bridge_dirichlet(N, alpha_weights):
    ## COMPUTE L for the multivariate gaussian used to represent the dirichlet on the log space (Laplace bridge) 
    
    try:
        Ni = Integer(N)
        M = Matrix([[Integer(i==j)/alpha_weights - 1/(Ni*alpha_weights) for j in range(Ni)] for i in range(Ni)])
        L = M.cholesky(hermitian=False)
        L = jnp.array(np.array(L).astype(np.float64))
    except: 
        M = np.array([[int(i==j)/alpha_weights - 1/(N*alpha_weights) for j in range(N)] for i in range(N)])
        try:
            L = scipy.linalg.cholesky(M, lower=True)
        except:
            L = icholesky(M, lower=True)
        L = jnp.array(L)

    return L

def norm_to_logdirichlet(logits, L):
    if False: ## test for rebuttal
        logweights = jax.nn.log_softmax(logits)
    elif False: # this ignores non-diagonal elements
        logweights = nn.log_softmax(logits * jnp.sqrt((N-1)/(alpha_weights * N)))
    else:
        logweights = jax.nn.log_softmax(L@logits)
    return logweights

#EXP_SCALE = True

def norm_to_ab(logits_s,inv_lambda, logits_l, alpha_quant, L= None):
    if True:
        unif = jax.scipy.stats.norm.cdf(logits_s)
        expon =  expon_inverse_cdf(unif)#inverse cdf of exponential distribution
        s = expon * inv_lambda
    else:
        s = jax.nn.softplus(logits_s + inv_lambda) 
        #s = jnp.exp(logits_s*0.1 + jnp.log(8000)) # jsut a quick test to see how it changes

    if L is not None: # sorted locations
        ## ASSUMING L makes dirichlet(1,..,1)
        dirichlet = jax.nn.softmax(L@logits_l)
        # sorted uniforms
        unif = jnp.cumsum(dirichlet)[:-1]  # remove 1 to get N sorted uniform variables

    else:
        unif = jax.scipy.stats.norm.cdf(logits_l)


    if alpha_quant is not None:
        s = soft_round(s, alpha = alpha_quant)
        # unif on range [-0.5, s + 0.5] 

        l  = unif * (s+1) - 0.5
        l = soft_round(l, alpha = alpha_quant) 
    else:
        l = unif * s 

    a = l + 1 
    b = s - l + 1

    return a,b




def build_model1(Y_train, n_squeeze, input_size, activation, layers, nodes, n_segments, prior_sigma, prior_alpha_weights, prior_scale_ab, alpha_quant, STAX, parameterization, sorted_locations=False):

    #### UNCONDITIONAL PART

    cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate = train_gaussmix_squeeze(Y_train, n_squeeze)

    #### CONDITIONAL PART
    output_size = n_segments * 3   


    if sorted_locations:
        L1 = get_L_bridge_dirichlet(n_segments+1,1.)
        output_size += 1 
    else:
        L1 = None


    pred_normal_logits, init_fn, init_fns, get_output_from_last_hidden = build_pred_logits(activation, prior_sigma, nodes, layers, output_size, input_size, STAX, parameterization)


    def pred_betamix_uncalibrated(params, x, N, location=0, scale=1., alpha_weights=1., scale_ab=1.):
        logits = pred_normal_logits(params,x) 

        # exp as in standard gaussian density netwroks. could be softplus too
        alphas = jnp.exp(logits[:N]) + 1. # add 1. to define a family whose space of natural parameters is a cone 
        betas = jnp.exp(logits[:N]) + 1. # add 1. to define a family whose space of natural parameters is a cone 
        logweights = jax.nn.log_softmax(logits[(2*N):])
        
        return BetaMix(logweights=logweights, alphas=alphas, betas=betas, corr=None) #, shape=logweights.shape)



    def pred_betamix(params, x, return_probits, N, scale_ab, L, L1):  #location=0, scale=1., 
        ## NORMALIZATION
        logits = pred_normal_logits(params,x)

        ## SPLITTING
        size_s = N
        logits_s = logits[:(size_s)]

        if L1 is not None: #sorted locations
            size_l = N+1
        else:
            size_l = N
        

        logits_l = logits[(size_s):(size_s+size_l)]


        a,b = norm_to_ab(logits_s, scale_ab, logits_l, alpha_quant, L1)

        size_w = N
        logits_weight = logits[(size_s+size_l):(size_s+size_l+size_w)]

        logweights = norm_to_logdirichlet(logits_weight, L)

        if return_probits:
            return BetaMix(logweights=logweights, alphas=a, betas=b, corr=None), logits 
        else:
            return BetaMix(logweights=logweights, alphas=a, betas=b, corr=None) #, logits_before, logits

    if STAX:
        rng_key = random.PRNGKey(1234) # just to get the structure
        _,params_structure = init_fn(rng_key,(-1, input_size))
    else:
        sizes = [input_size] +[nodes]*layers + [output_size]
        params_structure = get_param_structure(sizes)


    L = get_L_bridge_dirichlet(n_segments, prior_alpha_weights)

    predict = partial(pred_betamix, return_probits=False, N = n_segments, scale_ab=prior_scale_ab, L=L, L1=L1) #scale=None, location=None, 
    predict_with_probits = partial(pred_betamix, return_probits=True, N = n_segments, scale_ab=prior_scale_ab, L=L, L1=L1) #scale=None, location=None, 




    # density of y given by the beta mixture dist
    # for one beta mixture, one y
    def logDens(y, dist):


        # one or more y, one weighted beta dist
        def weightedBetaLogPdf(y,dist):
            return jax.scipy.stats.beta.logpdf(y, dist.alphas, dist.betas) + dist.logweights     

        # vmap over beta dists
        vWeightedBetasLogPdf = vmap(weightedBetaLogPdf, (None,0))

        logdens = vWeightedBetasLogPdf(y,dist)
        logdens = jax.scipy.special.logsumexp(logdens, axis=0)

        return logdens

    def loglikelihood(params, X, y):

        y_squeezed = jnp.squeeze(cdf_squeeze(jnp.expand_dims(y,0), coord=0))
        #y_squeezed2 = jnp.squeeze(cdf_gaussmix1(y))
        #jax.debug.print("diff {diff} ", diff=y_squeezed-y_squeezed2)

        dist = predict(params, X)

        if False:
            logdens = jax.scipy.stats.beta.logpdf(y_squeezed,dist.alphas,dist.betas)
            logweightdens = logdens + dist.logweights

            # ll = jnp.log(jnp.sum(jnp.exp(logweightdens)))
            ll = jax.scipy.special.logsumexp(logweightdens)
            

        else:
            ll = logDens(y_squeezed, dist)
        #jax.debug.print("ll {ll} ll2 {ll2} ", ll=ll, ll2=ll2)

        if True: #UNSQUEEZE:
            logpdf_y = logpdf_unsqueeze(y, coord=0)
            ll = ll + jnp.squeeze(logpdf_y)

        return ll


    ##  mix M beta mixtures (over first dimension)
    def mix_preds_betamix(preds):

        M = preds.logweights.shape[0]
        logweights = rearrange(preds.logweights, 'b c -> (b c)') - jnp.log(M)
        alphas = rearrange(preds.alphas, 'b c -> (b c)')
        betas = rearrange(preds.betas, 'b c -> (b c)') 

        out = BetaMix(logweights=logweights,alphas=alphas,betas=betas, corr=None) 
        return out


    return predict, predict_with_probits, params_structure, cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate, loglikelihood, logDens, mix_preds_betamix, init_fn, init_fns
    

#import tensorflow as tf
#import tensorflow_probability as tfp
from tensorflow_probability.substrates import jax as tfp
from logbetacdf import log_beta_cdf_rect
from qnorm import qnorm5, qnorm_trick, qnorm_logit


def icholesky(H, lower):
    w, v = jnp.linalg.eigh(H)
    w = jnp.where(w < 0, 0.0001, w) # make this pd, psd is insufficient
    H_pd = v @ jnp.eye(H.shape[0])*w @ v.T

    return jax.scipy.linalg.cholesky(H_pd, lower=lower)

import jax._src.scipy.sparse.linalg

from numpyro.distributions.util import (
    matrix_to_tril_vec,
    signed_stick_breaking_tril,
    add_diag,
    vec_to_tril_matrix
)


### THIS commented BLOCK faisl on AZURE but works on local 

from oryx.core import inverse_and_ildj
from oryx.core import custom_inverse

@custom_inverse
def inverse_sigmoid(x):
  return jnp.log(x) - jnp.log(1. - x)
inverse_sigmoid.def_inverse_unary(jax.nn.sigmoid)  # Define  custom inverse.

import haiku as hk

class Affine(hk.Module):
  """A shift and scale layer."""
  def __init__(self, flip: bool = False, last=False):
    super().__init__()
    self.flip = flip  # Permute lower and upper blocks.
    self.last = last

  def __call__(self, z):
    """Do affine transformation on lower block."""


    # Make a 50:50 split.
    z_1, z_2 = jnp.split(z, 2)
    if self.flip:
      z_1, z_2 = z_2, z_1

    # Predict shift and scale parameters using neural network.
    n_features = z.shape[-1]
    neural_net = hk.nets.MLP(output_sizes=[512, 128, n_features], activation=jax.nn.relu)
    shift_scale = neural_net(z_1)
    shift, scale = jnp.split(shift_scale, 2, axis=-1)

    # Apply shift and scale transformation to lower block.
    x_1 = z_1
    x_2 = z_2 * jnp.exp(scale) + shift

    res = jnp.concatenate([x_1, x_2]) 
    if self.last:
        res = jax.nn.sigmoid(res)

    return res
  
@hk.without_apply_rng
@hk.transform
def forward(z):
  flow = hk.Sequential(layers=[
      Affine(),
      Affine(flip=True),
      Affine(),
      Affine(flip=True, last=True),
  ])
  return flow(z)


class GaussMix(typing.NamedTuple):
    logweights: jnp.ndarray
    mus: jnp.ndarray
    covs: jnp.ndarray

## n_squeeze is just for the unconditional
def  build_gaussian_modelD(with_scaling, Y_train, n_squeeze, input_size, activation, hidden_layers, nodes, N, prior_sigma, STAX, stax_parameterization, mean_y, std_y):
    D = Y_train.shape[1]

    #### UNCONDITIONAL PART : NOT NEEDED HERE, JUST TO KEEP THE SAME STRUCTURE AS BETAMIX
    _, _, logdens_gaussmix_multivariate = train_gaussmix_squeeze(Y_train, n_squeeze)


    def cdf_squeeze_linear(y):
        return (y - mean_y)/std_y
    
    def logpdf_unsqueeze_linear(y):
        assert(std_y.shape[0]==D)
        return jnp.sum(-jnp.log(std_y), axis=0) # sum over D
    
    cdf_squeeze = cdf_squeeze_linear
    logpdf_unsqueeze = logpdf_unsqueeze_linear


    #### CONDITIONAL PART

    output_size = N + 2*N*D  #diagonal cov  


    pred_logits, init_fn, init_fns, get_output_from_last_hidden = build_pred_logits(activation, prior_sigma, nodes, hidden_layers, output_size, input_size, STAX, stax_parameterization, with_scaling)

    def pred_gaussmix(params, x, return_probits, N, D):  
        logits = pred_logits(params,x)

        logweights = jax.nn.log_softmax(logits[:N])
        mus = logits[N:(N+N*D)]
        covs = jax.nn.softplus(logits[(N+N*D):])  ## diagonal cov matrix

        mus = mus.reshape(N,D)
        covs = covs.reshape(N,D)

        if return_probits:
            return GaussMix(logweights=logweights, mus=mus, covs=covs), logits 
        else:
            return GaussMix(logweights=logweights, mus=mus, covs=covs)

    if STAX:
        rng_key = random.PRNGKey(1234) # just to get the structure
        _,params_structure = init_fn(rng_key,(-1, input_size))
    else:
        sizes = [input_size] +[nodes]*hidden_layers + [output_size]
        params_structure = get_param_structure(sizes)

    predict = partial(pred_gaussmix, return_probits=False, N = N, D = Y_train.shape[1]) 
    predict_with_probits = partial(pred_gaussmix, return_probits=True, N = N, D = Y_train.shape[1]) 




    def logDens(y, dist):

        # one or more y, one weighted gaussian dist
        def weightedNormLogPdf(y,dist):
            return jax.scipy.stats.multivariate_normal.logpdf(jnp.expand_dims(y,0), dist.mus, jnp.diag(dist.covs)) + dist.logweights     

        # vmap over gauss components
        vWeightedNormLogPdf = vmap(weightedNormLogPdf, (None,0))

        logdens = vWeightedNormLogPdf(y,dist)
        logdens = jnp.squeeze(logdens)
        logdens = jax.scipy.special.logsumexp(logdens, axis=0)

        return logdens

    def logDens_uncorr(y, dist, covariate=None): 

        if covariate is not None:
            dist = GaussMix(mus=dist.mus[:,covariate], covs=dist.covs[:,covariate:(covariate+1)], logweights=dist.logweights)

        return jnp.squeeze(logDens(y, dist))


    def loglikelihood(params, XorPreds, y, logDens, from_preds):
        y_squeezed = cdf_squeeze(y)

        if from_preds:
            dist = XorPreds
        else:
            dist = predict(params, XorPreds)

        ll = logDens(y_squeezed, dist)

        if True: #UNSQUEEZE:

            logpdf_y = logpdf_unsqueeze(y)
    
            ll = ll + logpdf_y

        return ll





    ##  mix M beta mixtures (over first dimension)
    def mix_preds_gaussmix(preds):

        M = preds.logweights.shape[0]
        logweights = rearrange(preds.logweights, 'b c -> (b c)') - jnp.log(M)
        mus = rearrange(preds.mus, 'b c d -> (b c) d')
        covs = rearrange(preds.covs, 'b c d -> (b c) d') 

        out = GaussMix(logweights=logweights,mus=mus,covs=covs) 
        return out

    return predict, predict_with_probits, params_structure, cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate, loglikelihood, logDens, mix_preds_gaussmix, logDens_uncorr,init_fn, init_fns, get_output_from_last_hidden
    





    return predict, predict_with_probits, params_structure,cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate, loglikelihood, logDens, mix_preds_betamix, log_dens_uncorr, init_fn, init_fns, get_outputs_from_last_hidden



def  build_modelD(Y_train, n_squeeze, input_size, activation, hidden_layers, nodes, N, prior_sigma, prior_alpha_weights, prior_scale_ab, prior_corr_concentration, alpha_quant, STAX, parameterization, COPULA=True, bridgebetas=True, sorted_locations=False, lkj="onion", logbetacdf_N=1000, CHOLCORR=True, SQUEEZE="gaussmix", mean_y=None, std_y=None):

    D = Y_train.shape[1]

    NU = D

    #### UNCONDITIONAL PART
    cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate = train_gaussmix_squeeze(Y_train, n_squeeze)

    init_diffeo = None
    if SQUEEZE=="linear": #overwrite squeeze/unsqueeze
        def cdf_squeeze_linear(y):
            return (y - mean_y)/std_y
        
        def logpdf_unsqueeze_linear(y):
            return jnp.sum(-jnp.log(std_y), axis=0)
        
        cdf_squeeze = cdf_squeeze_linear
        logpdf_unsqueeze = logpdf_unsqueeze_linear

    elif SQUEEZE=="nn":
        #overwrite squeeze/unsqueeze

        if False:
            from neural_tangents import stax
            #from jax.example_libraries import stax

            activ = activation[0].upper() + activation[1:]

            if parameterization == 'standard_custom':
                param = 'standard'
                layers = [MyDense(nodes, W_std= prior_sigma*jnp.sqrt(D)  , b_std= prior_sigma , parameterization=param),eval("stax." + activ)() ]
                layers += [MyDense(nodes, W_std= prior_sigma*jnp.sqrt(nodes)  , b_std= prior_sigma , parameterization=param),eval("stax." + activ)() ]*(hidden_layers-1) 
                layers += [MyDense(D, W_std= prior_sigma*jnp.sqrt(nodes), b_std= prior_sigma  , parameterization=param)]
            else:
                layers = [MyDense(nodes, W_std= prior_sigma  , b_std= prior_sigma , parameterization=param),eval("stax." + activ)() ]*hidden_layers + [ MyDense(D, W_std= prior_sigma, b_std= prior_sigma  , parameterization=param)]

            #output in [0,1]^D
            layers += [stax.Sigmoid_like()]

            init_diffeo, apply_diffeo, _ = stax.serial(*layers)


            def cdf_squeeze_diffeo(y, params):
                y_linear_squeezed = (y - min_y)/(max_y - min_y)
                return apply_diffeo(params,y_linear_squeezed)
            
            def logpdf_unsqueeze_diffeo(y, params):
                linear_unsqueeze = jnp.sum(-jnp.log(max_y - min_y), axis=0)
                return linear_unsqueeze + jnp.linalg.slogdet(jax.jacobian(apply_diffeo,1)(params,y))[1]  # take jacobian wrt y

            cdf_squeeze = cdf_squeeze_diffeo
            logpdf_unsqueeze = logpdf_unsqueeze_diffeo

        else:

            # Make f(z) invertible by combining params and forward.apply.
            f_inv_and_ildj = inverse_and_ildj(lambda p, z: [p, forward.apply(p, z)])

            # def log_prob(params, x):
            #     """Log-likelihood of a single example x."""
            #     (_, z), inv_log_det_jac = f_inv_and_ildj(params, x)
            #     logp_z = jax.scipy.stats.norm.logpdf(z)
            #     logp_x = logp_z.sum() + inv_log_det_jac
            #     return logp_x

            LINEAR_SQUEEZE = True
            def cdf_squeeze_diffeo(y, params):
                if LINEAR_SQUEEZE:
                    y_linear_squeezed = (y - mean_y)/(std_y)
                    y_squeezed = forward.apply(params, y_linear_squeezed)
                else:
                    y_squeezed = forward.apply(params, y)
                return y_squeezed


            def logpdf_unsqueeze_diffeo(y, params):
                if LINEAR_SQUEEZE:
                    linear_unsqueeze = jnp.sum(-jnp.log(std_y), axis=0)
                    y_linear_squeezed = (y - mean_y)/(std_y)
                    y_squeezed = forward.apply(params, y_linear_squeezed)
                else:
                    y_squeezed = forward.apply(params, y)
                    linear_unsqueeze = 0.

                # Compute: z = f⁻¹(x) and ln |det ∂f⁻¹/∂x|.
                (_, y_), inv_log_det_jac = f_inv_and_ildj(params,y_squeezed)
                return -inv_log_det_jac + linear_unsqueeze
                #return linear_unsqueeze + jnp.linalg.slogdet(jax.jacobian(forward.apply,1)(params,y))[1]

            cdf_squeeze = cdf_squeeze_diffeo
            logpdf_unsqueeze = logpdf_unsqueeze_diffeo

            
            init_diffeo = forward.init #Y_train[0] to tell the shape


    #### CONDITIONAL PART

    if CHOLCORR:
        output_size = N * (1 + 2*D +  (D*(D-1)//2))  
        if lkj=="onion":
            output_size += N

    else:
        output_size = N + 2*N*D + N*NU*D 

    if sorted_locations:
        L1 = get_L_bridge_dirichlet(N+1,1.)
        output_size += D 
    else:
        L1 = None

    pred_normal_logits, init_fn, init_fns, get_output_from_last_hidden = build_pred_logits(activation, prior_sigma, nodes, hidden_layers, output_size, input_size, STAX, parameterization)

    def cov_to_corr(cov):

        Dinv = jnp.diag(1 / jnp.sqrt(jnp.diag(cov))) 
        corr = Dinv @ cov @ Dinv
        return corr

    def to_corr_wishart(logits_cov, dimension, concentration=1.):
        if True: ## parameter of the prior, just take Identity
            G = logits_cov
        else:
            V = jnp.eye(D) ## parameter of the prior, just take Identity
            B = jnp.linalg.cholesky(V) 
            G = B@logits_cov
        cov = G@ jnp.transpose(G,(1,0))
        
        if False:
            corr = cov_to_corr(cov) 
        
        else: # Maurizio's suggestion
            Dinv = jnp.diag(1 / jnp.sqrt(jnp.diag(cov))) 
            normG = (Dinv @ G) 
            corr = normG @ jnp.transpose(normG,(1,0))

        return corr

    def to_corr_gram(logits_cov): #Random Gram Method
        safe_normalize_columns = vmap(jax._src.scipy.sparse.linalg._safe_normalize, in_axes=1, out_axes=(1,0)) # for one matrix m x n, normalize columns
        u,_ = safe_normalize_columns(logits_cov)  # random points on the unit sphere
        corr = u.T @ u
        return corr


    # equivalent to tfp: 
    def to_gram_cholcorr(probits): 
        normal_sample = tfp.math.fill_triangular(probits) # looks equivalent

        cholesky = jnp.zeros((D,D))
        cholesky = cholesky.at[..., 1:, :-1].set(normal_sample)

        cholesky = cholesky + jnp.identity(cholesky.shape[0])


        # The following Normal distribution is used to create a uniform distribution on
        # a hypershere (ref: http://mathworld.wolfram.com/HyperspherePointPicking.html)
        cholesky /= jnp.linalg.norm(cholesky, axis=-1, keepdims=True)  # dangerous => replace by safe_norm

        #safe_normalize_rows = vmap(jax._src.scipy.sparse.linalg._safe_normalize) # for one matrix m x n, normalize rows
        #u_hypershere,_ = safe_normalize_rows(normal_sample)  # random points on the unit sphere

        # correct the diagonal
        # NB: we clip due to numerical precision
        #diag = jnp.sqrt(jnp.clip(1 - jnp.sum(cholesky ** 2, axis=-1), a_min=0.))
        #cholesky = cholesky + jnp.expand_dims(diag, axis=-1) * jnp.identity(cholesky.shape[0])

        return cholesky


    def to_cholcorr(probits, dimension, concentration=1.): 
        return tfp.bijectors.CorrelationCholesky().forward(probits)

    def to_beta(probits, a, b): 
        u = jax.scipy.stats.norm.cdf(probits)  
        #jax.debug.print("a {a} b {b} u {u}", a=a, b=b, u=u)
        return tfp.math.betaincinv(a, b, u)



    # from https://github.com/mariushobbhahn/LB_for_BNNs_official/tree/bf003ed104407d0cfd6cc873d2867b80f5888aa5/fig_2D_LB.ipynb
    def beta_t_logit_mu(a,b):
        return(jnp.log(a/b))

    def beta_t_logit_var(a,b):
        return((a+b)/(a*b))

    def logistic_transform(logit):
        return(1 / (1 + jnp.exp(-logit)))



    #CVINE adapted from numpyro
    def to_LKJCholesky_cvine(probits, dimension, concentration=1.):

        # We construct base distributions to generate samples for each method.
        # The purpose of this base distribution is to generate a distribution for
        # correlation matrices which is propotional to `det(M)^{\eta - 1}`.
        # (note that this is not a unique way to define base distribution)
        # Both of the following methods have marginal distribution of each off-diagonal
        # element of sampled correlation matrices is Beta(eta + (D-2) / 2, eta + (D-2) / 2)
        # (up to a linear transform: x -> 2x - 1)
        Dm1 = dimension - 1
        marginal_concentration = concentration + 0.5 * (dimension - 2)
        offset = 0.5 * jnp.arange(Dm1)

        ## CVINE method
        ### adapted from NumPyro 

        # The following construction follows from the algorithm in Section 2.4 of [1]:
        # offset_tril is [0, 1, 1, 2, 2, 2,...] / 2
        offset_tril = matrix_to_tril_vec(jnp.broadcast_to(offset, (Dm1, Dm1)))
        beta_concentration = (
            jnp.expand_dims(marginal_concentration, axis=-1) - offset_tril
        )

        #self._beta = Beta(beta_concentration, beta_concentration)
        #beta_sample = self._beta.sample(key, size)
        if not bridgebetas: #this gives Nans sometimes
            beta_sample = to_beta(probits[0], beta_concentration, beta_concentration)
        else:
            mu = beta_t_logit_mu(beta_concentration,beta_concentration)
            sigma = jnp.sqrt(beta_t_logit_var(beta_concentration,beta_concentration))
            beta_sample = logistic_transform(probits[:3] * sigma + mu) 

        partial_correlation = 2 * beta_sample - 1  # scale to domain to (-1, 1)
        return signed_stick_breaking_tril(partial_correlation)


    # ONION: adapted from numpyro
    def to_LKJCholesky_onion(probits, dimension, concentration=1.):

        # We construct base distributions to generate samples for each method.
        # The purpose of this base distribution is to generate a distribution for
        # correlation matrices which is propotional to `det(M)^{\eta - 1}`.
        # (note that this is not a unique way to define base distribution)
        # Both of the following methods have marginal distribution of each off-diagonal
        # element of sampled correlation matrices is Beta(eta + (D-2) / 2, eta + (D-2) / 2)
        # (up to a linear transform: x -> 2x - 1)
        Dm1 = dimension - 1
        marginal_concentration = concentration + 0.5 * (dimension - 2)
        offset = 0.5 * jnp.arange(Dm1)

        ## ONION method
        ### adapted from NumPyro 

        # The following construction follows from the algorithm in Section 3.2 of [1]:
        # NB: in [1], the method for case k > 1 can also work for the case k = 1.
        beta_concentration0 = jnp.expand_dims(marginal_concentration, axis=-1) - offset
        beta_concentration1 = offset + 0.5


        # Now we generate w term in Algorithm 3.2 of [1].
        if not bridgebetas:  #this gives Nans sometimes
            beta_sample = to_beta(probits[0], beta_concentration1, beta_concentration0)
        else:
            mu = beta_t_logit_mu(beta_concentration1,beta_concentration0)
            sigma = jnp.sqrt(beta_t_logit_var(beta_concentration1,beta_concentration0))
            beta_sample = logistic_transform(probits[0] * sigma + mu) 

        normal_sample = probits[1:]


        #normal_sample = tfp.math.fill_triangular(normal_sample) # looks equivalent
        normal_sample = vec_to_tril_matrix(normal_sample, diagonal=0)

        # The following Normal distribution is used to create a uniform distribution on
        # a hypershere (ref: http://mathworld.wolfram.com/HyperspherePointPicking.html)
        u_hypershere = normal_sample / jnp.linalg.norm(normal_sample, axis=-1, keepdims=True)  # dangerous => replace by safe_norm

        #safe_normalize_rows = vmap(jax._src.scipy.sparse.linalg._safe_normalize) # for one matrix m x n, normalize rows
        #u_hypershere,_ = safe_normalize_rows(normal_sample)  # random points on the unit sphere



        w = jnp.expand_dims(jnp.sqrt(beta_sample), axis=-1) * u_hypershere

        if False:
            # put w into the off-diagonal triangular part
            cholesky = jnp.zeros((dimension,dimension))
            cholesky = cholesky.at[..., 1:, :-1].set(w)
            # equivalent
            #cholesky = jnp.concatenate([jnp.concatenate([jnp.zeros((1,Dm1)),w]),jnp.zeros((dimension,1))],axis=1)

            # correct the diagonal
            # NB: we clip due to numerical precision
            diag = jnp.sqrt(jnp.clip(1 - jnp.sum(cholesky ** 2, axis=-1), a_min=0.))
            cholesky = cholesky + jnp.expand_dims(diag, axis=-1) * jnp.identity(dimension)
        else:
            # put w into the off-diagonal triangular part
            cholesky = jnp.zeros((dimension,dimension))
            cholesky = cholesky.at[..., 1:, :-1].set(w)
            # correct the diagonal
            # NB: beta_sample = sum(w ** 2) because norm 2 of u is 1.
            diag = jnp.ones(cholesky.shape[:-1]).at[..., 1:].set(jnp.sqrt(1 - beta_sample))
            cholesky = add_diag(cholesky, diag)

        return cholesky

    if lkj=="onion":
        to_LKJCholesky = to_LKJCholesky_onion
    elif lkj=="cvine":
        to_LKJCholesky = to_LKJCholesky_cvine
    elif lkj=="norm":
        to_LKJCholesky = to_cholcorr
    else:
        exit(1)
    to_LKJCholesky = partial(to_LKJCholesky, concentration=prior_corr_concentration)



    def pred_betamix(params, x, return_probits, N, D, NU, scale_ab, L, L1):  
        ## NORMALIZATION
        logits = pred_normal_logits(params,x)

        ## SPLITTING
        size_w = N
        logits_weight = logits[:size_w]
        size_s = N*D
        logits_s = logits[size_w:(size_w+size_s)]  #+1 for each dimension

        if L1 is not None: #sorted locations
            size_l = N*D +D
        else:
            size_l = N*D

        logits_l = logits[(size_w+size_s):(size_w+size_s + size_l)]
        
        if CHOLCORR:
            logits_cov = logits[(size_w+size_s+size_l):].reshape((N,-1))
        else:
            logits_cov = logits[(size_w+size_s+size_l):].reshape((N,NU,D))

        ## DISTRIBUTION TRANSFORMATION
        if CHOLCORR:
            corr = vmap(to_LKJCholesky, (0,None))(logits_cov, D)
            #corr = vmap(to_gram_cholcorr)(logits_cov)
        else:
            corr = vmap(to_corr_gram)(logits_cov)

        logweights = norm_to_logdirichlet(logits_weight, L)


        if L1 is not None: #sorted locations
            logits_s = logits_s.reshape(N,-1)
            logits_l = logits_l.reshape(N+1,-1)
            a,b = vmap(norm_to_ab, in_axes=(1, None, 1, None, None), out_axes=(1,1))(logits_s, scale_ab, logits_l, alpha_quant, L1)  # for each target dimension
        else:
            a,b = norm_to_ab(logits_s, scale_ab, logits_l, alpha_quant, L1)
            a = a.reshape(N,D)
            b = b.reshape(N,D)

        if return_probits:
            return BetaMix(logweights=logweights, alphas=a, betas=b, corr=corr), logits 
        else:
            return BetaMix(logweights=logweights, alphas=a, betas=b, corr=corr) 


    if STAX:
        rng_key = random.PRNGKey(1234) # just to get the structure
        _,params_structure = init_fn(rng_key,(-1, input_size))
    else:
        sizes = [input_size] +[nodes]*hidden_layers + [output_size]
        params_structure = get_param_structure(sizes)

    L = get_L_bridge_dirichlet(N, prior_alpha_weights)

    predict = partial(pred_betamix, return_probits=False, N = N, D = Y_train.shape[1], NU = Y_train.shape[1], scale_ab=prior_scale_ab, L=L, L1=L1) #scale=None, location=None, 
    predict_with_probits = partial(pred_betamix, return_probits=True, N = N, D = Y_train.shape[1], NU = Y_train.shape[1], scale_ab=prior_scale_ab, L=L, L1=L1) #scale=None, location=None, 


    # density of y given by the beta mixture dist
    # for one beta mixture, one y
    def logDens(y, dist):

        # one or more y, one weighted beta dist
        def weightedCopulaBetaLogPdf(y,dist):

            if COPULA: #use copula structure

                NO_APPROX = False
                if NO_APPROX: #Nans
                    #u = jax.scipy.stats.beta.cdf(y, dist.alphas, dist.betas) #gradients not implemented
                    #u = jax.lax.betainc(dist.alphas, dist.betas, y) #same problem, gradients not implemented
                    #u = tf.math.betainc(dist.alphas, dist.betas, y)

                    u = tfp.math.betainc(dist.alphas, dist.betas, y) ## if alphas and betas are big, can give 0 1 values then z is inf and nans later 
                    #log_u = jnp.log(u)
                    z = jax.scipy.stats.norm.ppf(u)

                else: #rectangle rule approximation to get log
                    #log_u_trap = vmap(log_beta_cdf_trap)(y, dist.alphas, dist.betas)

                    try:
                        log_beta_cdf_rect_N = partial(log_beta_cdf_rect,N=logbetacdf_N)
                        log_u_rect = vmap(log_beta_cdf_rect_N)(y, dist.alphas, dist.betas)
                    except FloatingPointError:
                        jax.debug.print("alphas {alphas} betas {betas} ", alphas=dist.alphas, beta=dist.betas)
                        exit() 
                    
                    log_u = jnp.where(log_u_rect>=0.,-1e-10,log_u_rect)
                

                    #log_u = jnp.minimum(log_u_trap, log_u_rect)
                    #log_u = jnp.where(log_u_rect>=0.,-jnp.finfo(float).smallest_normal,log_u_rect)
                    
                    #log_u = vmap(logpbeta)(y, dist.alphas, dist.betas) ## doesnt work

                    #u = y
                    #jax.debug.print("log_u_trap {log_u_trap} log_u_rect {log_u_rect} ", log_u_trap=log_u_trap, log_u_rect=log_u_rect)
                    #z = jax.scipy.stats.norm.ppf(jnp.exp(log_u))

                    try: 
                        z = qnorm5(log_u, mu=0., sigma=1., lower_tail=True, log_p=True)
                        #z = qnorm_trick(log_u) #### THIS INTRODUCES THE WEIRD "FOLDING" ARTIFACTS
                    except FloatingPointError:
                        jax.debug.print("log_u {log_u} ", log_u=log_u)
                        exit() 

                    #z = qnorm_logit(log_u)
                    #jax.debug.print("z {z} z2 {z2}", z=z, z2=z2)
                    #jax.debug.print("y {y} alpha {alpha} beta {beta} log_u {log_u} log_of_u {log_of_u} z {z} z_scipy_exp_log_u {z_scipy}",y=y, log_u=log_u, log_of_u=jnp.log(u), z=z, z_scipy=z_scipy, alpha=dist.alphas, beta=dist.betas)
                    #jax.debug.print("y {y} alpha {alpha} beta {beta} log_u {log_u} z {z}",y=y, log_u=log_u,  z=z, alpha=dist.alphas, beta=dist.betas)
                



                #copula = 1./jnp.sqrt(jnp.linalg.det(dist.corr)) * jnp.exp(-.5 * z @ (jnp.linalg.inv(dist.corr)-jnp.eye(N=dist.corr.shape[0])) @ z )


                if False:
                    inv_corr = jnp.linalg.inv(dist.corr)
                    quad_form = z @ (inv_corr-jnp.eye(N=dist.corr.shape[0])) @ z
                    #quad_form2 = z @ inv_corr @ z  - z @ z

                    #jax.debug.print("diff_inv {diff} ", diff=inv_L - jnp.linalg.inv(L))      
                
                else:

                    try:
                        #L = icholesky(dist.corr, lower=True)
                        if CHOLCORR:
                            L = dist.corr ## dist.corr in cholesky space already
                        else:
                            #L = jnp.linalg.cholesky(dist.corr)
                            L = icholesky(dist.corr, lower=True)

                    except FloatingPointError:
                        jax.debug.print("corr {corr} ", corr=dist.corr)
                        exit()
                    #inv_L2 = jax.scipy.linalg.solve_triangular(L2, jnp.identity(L2.shape[1]), lower=True)
                    inv_L = jax.scipy.linalg.solve_triangular(L, jnp.identity(L.shape[1]), lower=True)


                    #jax.debug.print("inv_L {inv_L} L {L}", inv_L=inv_L, L=L)      

                    try:
                        quad_form = jnp.linalg.norm(inv_L @ z, ord=2)**2  - z @ z
                    except FloatingPointError:
                        jax.debug.print("inv_L {inv_L} z {z}", inv_L=inv_L, z=z)
                        exit()
                #jax.debug.print("diff {diff} ", diff=quad_form - quad_form)      

                #

                #jax.debug.print("diag_L {diag_L} ", diag_L=jnp.diagonal(L)) 

                try:
                    
                    #log_det_bad = jnp.linalg.slogdet(dist.corr @ dist.corr.T)[1]
                    #https://math.stackexchange.com/questions/3158303/using-cholesky-decomposition-to-compute-covariance-matrix-determinant
                    log_det = 2. * jnp.sum(jnp.log(jnp.diagonal(L)))

                    #jax.debug.print("diff {diff} ", diff=log_det - log_det_bad)

                    log_copula = -.5 * (log_det +  quad_form )
                    
                except FloatingPointError:
                        jax.debug.print("quad_form {quad_form} corr {corr}", quad_form=quad_form, corr=dist.corr)
                        exit()
                #log_copula = -.5 * (jnp.log(jnp.linalg.det(dist.corr)) +  quad_form )

                #jax.debug.print("diff {diff} ", diff=log_copula - log_copula2)
                #jax.debug.print("y {y} alpha {alpha} beta {beta} log_u {log_u} z {z}  log_copula {log_copula}",y=y, log_u=log_u, z=z, alpha=dist.alphas, beta=dist.betas, log_copula=log_copula )
                #jax.debug.print("y {y} alpha {alpha} beta {beta}  log_copula {log_copula} ", y=y, alpha=dist.alphas, beta=dist.betas, log_copula=log_copula )

                return jnp.sum(jax.scipy.stats.beta.logpdf(y, dist.alphas, dist.betas), axis=0) + dist.logweights + log_copula ##sum is over D
            else: # no copula structure
                return jnp.sum(jax.scipy.stats.beta.logpdf(y, dist.alphas, dist.betas), axis=0) + dist.logweights  ##sum is over D

        # vmap over components
        vWeightedBetasLogPdf = vmap(weightedCopulaBetaLogPdf, (None,0))

        logdens = vWeightedBetasLogPdf(y,dist)
        logdens = jax.scipy.special.logsumexp(logdens, axis=0)

        return logdens

    # no correlation in this version, just a beta mixture of uncorrelated betas (each component is a product of betas)
    # density of y given by the beta mixture dist
    # for one beta mixture, one y
    #def logDens_no_correlation(y, dist):
    def logDens_uncorr(y, dist, covariate=None): 

        if covariate is not None:
            dist = BetaMix(alphas=dist.alphas[:,covariate], betas=dist.betas[:,covariate], logweights=dist.logweights, corr=dist.corr)


        # one or more y, one weighted beta dist (=one component)
        def weightedBetaLogPdf(y,dist):
            # sum over y components (density is just product of marginal densities)
            #return jnp.sum(jax.scipy.stats.beta.logpdf(y, dist.alphas, dist.betas)) + dist.logweights     
            return jnp.sum(jax.scipy.stats.beta.logpdf(y, dist.alphas, dist.betas)) + dist.logweights     ## when only one covariate is used, sums over 1 element, otherwise sums over two

        # vmap over beta dists
        vWeightedBetasLogPdf = vmap(weightedBetaLogPdf, (None,0))

        logdens = vWeightedBetasLogPdf(y,dist)

        logdens = jax.scipy.special.logsumexp(logdens, axis=0) # sum over components

        return logdens



    def loglikelihood_old(params, XorPreds, y, logDens, from_preds):

        y_squeezed = jnp.squeeze(cdf_squeeze(jnp.expand_dims(y,0), coord=None),axis=(0,2))  #expand_dims to make a batch of one element  # I am not squeezing axis 1 because it's  the dimension of covariates and I want it to work for D=1
        #y_squeezed = cdf_squeeze(jnp.expand_dims(y,0), coord=0)

        if from_preds:
            dist = XorPreds
        else:
            dist = predict(params, XorPreds)

        ll = logDens(y_squeezed, dist)

        if True: #UNSQUEEZE:
            logpdf_y = logpdf_unsqueeze(y, coord=None)
            ll = ll + jnp.squeeze(logpdf_y)

        return ll


    def loglikelihood(params, XorPreds, y, logDens, from_preds, params_diffeo=None):

        if SQUEEZE=="nn":
            if params_diffeo is None:
                params_diffeo = params[0]
                params = params[1] # the rest, params_diffeo is params[0] because it is being trained
            y_squeezed = cdf_squeeze_diffeo(y, params=params_diffeo)
        else:
            y_squeezed = cdf_squeeze(y)

        if from_preds:
            dist = XorPreds
        else:
            dist = predict(params, XorPreds)

        ll = logDens(y_squeezed, dist)

        if True: #UNSQUEEZE:

            if SQUEEZE=="nn":
                logpdf_y = logpdf_unsqueeze_diffeo(y,params=params_diffeo)
            else:
                logpdf_y = logpdf_unsqueeze(y)
    
            ll = ll + logpdf_y

        return ll



    ##  mix M beta mixtures (over first dimension)
    def mix_preds_betamix(preds):

        M = preds.logweights.shape[0]
        logweights = rearrange(preds.logweights, 'b c -> (b c)') - jnp.log(M)
        alphas = rearrange(preds.alphas, 'b c d -> (b c) d')
        betas = rearrange(preds.betas, 'b c d -> (b c) d') 
        corr = rearrange(preds.corr, 'b c d1 d2 -> (b c) d1 d2') 

        out = BetaMix(logweights=logweights,alphas=alphas,betas=betas, corr=corr) 
        return out


    return predict, predict_with_probits, params_structure, cdf_squeeze, logpdf_unsqueeze, logdens_gaussmix_multivariate, loglikelihood, logDens, mix_preds_betamix, logDens_uncorr,init_fn, init_fns, init_diffeo, get_output_from_last_hidden
    


def logprior(nn_params,prior_params):
    logP = 0.0

    for ((w, b),(param_w,param_b)) in zip(nn_params,prior_params):
        logP += jnp.sum(jax.scipy.stats.norm.logpdf(w,loc=param_w[0] ,scale=param_w[1]))
        logP += jnp.sum(jax.scipy.stats.norm.logpdf(b,loc=param_b[0] ,scale=param_b[1]))
    return logP



