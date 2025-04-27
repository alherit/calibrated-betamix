import jax.example_libraries
import numpy as np
import jax
from jax import numpy as jnp
#import types 

import matplotlib
# no type 3 fonts
matplotlib.rcParams['pdf.fonttype'] = 42

from matplotlib import pyplot as plt
import matplotlib
import mlflow


def savefigs(fn,types=('.pdf','.png')):
    fig = plt.gcf()
    for t in types:
        fig.savefig(fn+t, bbox_inches='tight')

    plt.close()
    for t in types:
        mlflow.log_artifact(fn+t)
        os.remove(fn+t) #don't leave garbage


def get_loc_1(x):
    return 3.*np.sin(3*x+2)

def get_loc_2(x):
    return -2*(x<0)

# def get_loc_3(x):
#     return get_loc_2(x)


def get_scale_1(x):
    return .02

def get_scale_2(x):
    return .2

# def get_scale_3(x):
#    return get_scale_2(x)

def get_y1(x):
    return get_loc_1(x) + get_scale_1(x)*np.random.normal(size=x.shape)
    
def get_y2(x):
    return get_loc_2(x) + get_scale_2(x)*np.random.normal(size=x.shape)

# def get_y3(x):
#    return get_loc_3(x) + get_scale_3(x)*np.random.normal(size=x.shape)

def get_ps_truth(x,ys):
#    return (scipy.stats.norm.pdf(ys, get_loc_1(x),get_scale_1(x)) + scipy.stats.norm.pdf(ys, get_loc_2(x),get_scale_2(x)) + scipy.stats.norm.pdf(ys, get_loc_3(x), get_scale_3(x) )) / 3.
    return (scipy.stats.norm.pdf(ys, get_loc_1(x),get_scale_1(x)) + scipy.stats.norm.pdf(ys, get_loc_2(x),get_scale_2(x)) ) / 2.

def get_all(x, s):
#    return np.concatenate([get_y1(x[:s]),get_y2(x[s:(2*s)]), get_y3(x[(2*s):]) ])
    return np.concatenate([get_y1(x[:s]),get_y2(x[s:(2*s)]) ])



"""Adapted from pymc library"""
def gelman_rubin(x, return_var=False):
    """ Returns estimate of R for a set of traces.

    The Gelman-Rubin diagnostic tests for lack of convergence by comparing
    the variance between multiple chains to the variance within each chain.
    If convergence has been achieved, the between-chain and within-chain
    variances should be identical. To be most effective in detecting evidence
    for nonconvergence, each chain should have been initialized to starting
    values that are dispersed relative to the target distribution.

    Parameters
    ----------
    x : array-like
      An array containing the 2 or more traces of a stochastic parameter. That is, an array of dimension m x n x k, where m is the number of traces, n the number of samples, and k the dimension of the stochastic.
      
    return_var : bool
      Flag for returning the marginal posterior variance instead of R-hat (defaults of False).

    Returns
    -------
    Rhat : float
      Return the potential scale reduction factor, :math:`\hat{R}`

    Notes
    -----

    The diagnostic is computed by:

      .. math:: \hat{R} = \sqrt{\frac{\hat{V}}{W}}

    where :math:`W` is the within-chain variance and :math:`\hat{V}` is
    the posterior variance estimate for the pooled traces. This is the
    potential scale reduction factor, which converges to unity when each
    of the traces is a sample from the target posterior. Values greater
    than one indicate that one or more chains have not yet converged.

    References
    ----------
    Brooks and Gelman (1998)
    Gelman and Rubin (1992)"""

    if np.shape(x) < (2,):
        raise ValueError(
            'Gelman-Rubin diagnostic requires multiple chains of the same length.')

    try:
        m, n = np.shape(x)
    except ValueError:
        return [gelman_rubin(np.transpose(y)) for y in np.transpose(x)]

    # Calculate between-chain variance
    B_over_n = np.sum((np.mean(x, 1) - np.mean(x)) ** 2) / (m - 1)

    # Calculate within-chain variances
    W = np.sum(
        [(x[i] - xbar) ** 2 for i,
         xbar in enumerate(np.mean(x,
                                   1))]) / (m * (n - 1))

    # (over) estimate of variance
    s2 = W * (n - 1) / n + B_over_n
    
    if return_var:
        return s2

    # Pooled posterior variance estimate
    V = s2 + B_over_n / m

    # Calculate PSRF
    R = V / W

    return np.sqrt(R)


def compute_rhat_classification(samples, n_chains):
    """A wrapper function for computing R-hat statistics for regression tasks.

    Args:
        samples: numpy array [n_chains * n_samples, n_data, n_classes], the samples
            taken from MCMC sampling.
        n_chains: int, the number of sampling chains.

    Return:
        r_hat: numpy array [n_data * n_classes], the R hat statistics for each variable.
    """
    n_samples = samples.shape[0]
    n_vars = samples.shape[1] * samples.shape[2]
    samples = samples.reshape(n_samples, -1)
    samples = samples.reshape(n_chains, n_samples//n_chains, n_vars)
    r_hat = np.array(gelman_rubin(samples))
    return r_hat

def compute_rhat_regression(samples, n_chains):
    """A wrapper function for computing R-hat statistics for regression tasks.

    Args:
        samples: numpy array [n_chains * n_samples, n_vars], the samples
            taken from MCMC sampling.
        n_chains: int, the number of sampling chains.

    Return:
        r_hat: numpy array [n_vars], the R hat statistics for each variable.
    """
    n_samples = samples.shape[0]
    n_vars = samples.shape[1]
    samples = samples.reshape(n_chains, n_samples//n_chains, n_vars)
    r_hat = np.array(gelman_rubin(samples))
    return r_hat

#https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
from jax.tree_util import tree_flatten
def tree_stack(trees, axis=0):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l, axis=axis) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def tree_concat(trees, axis=0):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.concatenate(l, axis=axis) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None:
        treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def tree_random_normal(rng_key, target, batch_size):
    keys_tree = random_split_like_tree(rng_key, target)
    # return jax.tree_multimap(
    return jax.tree_util.tree_map(
        lambda mu_sigma, k: jax.random.normal(k,(batch_size,) + mu_sigma[0].shape) * mu_sigma[1] + mu_sigma[0],
        target,
        keys_tree,
    )


def tree_random_normal_single(rng_key, target):
    keys_tree = random_split_like_tree(rng_key, target)
    # return jax.tree_multimap(
    return jax.tree_util.tree_map(
        lambda mu_sigma, k: jax.random.normal(k, mu_sigma[0].shape) * mu_sigma[1] + mu_sigma[0],
        target,
        keys_tree,
    )



import functools
from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from jax import jit, lax, random
from tqdm.auto import tqdm

from sgmcmcjax.types import DiffusionState, PRNGKey, PyTree, SamplerState, SVRGState
from sgmcmcjax.util import progress_bar_scan

def _build_compiled_sampler(
    init_fn: Callable[[PRNGKey, PyTree], SamplerState],
    kernel: Callable[[int, PRNGKey, SamplerState], SamplerState],
    get_params: Callable[[SamplerState], PyTree],
    pbar: bool = True,
) -> Callable:
    """Build generic compiled sampler

    Args:
        init_fn (Callable[[PRNGKey, PyTree], SamplerState]): function to initialise the state of chain
        kernel (Callable[[int, PRNGKey, SamplerState], SamplerState]): transition kernel
        get_params (Callable[[SamplerState], PyTree]): functions that gets the target parameters from the state
        pbar (bool, optional): whether or not to display the progress bar. Defaults to True.

    Returns:
        Callable: sampling function with the same signature as kernel
    """

    @partial(jit, static_argnums=(1,))
    def sampler(key, Nsamples, state):
        def body(carry, i):
            key, state = carry
            key, subkey = random.split(key)
            state = kernel(i, subkey, state)
            return (key, state), get_params(state)

        key, subkey = random.split(key)
        #state = init_fn(subkey, params)

        lebody = progress_bar_scan(Nsamples)(body) if pbar else body
        (_, state), samples = lax.scan(lebody, (key, state), jnp.arange(Nsamples))
        return samples, state

    return sampler


def _build_noncompiled_sampler(
    init_fn: Callable[[PRNGKey, PyTree], SamplerState],
    kernel: Callable[[int, PRNGKey, SamplerState], SamplerState],
    get_params: Callable[[SamplerState], PyTree],
    pbar: bool = True,
) -> Callable:
    """Build generic non-compiled sampler

    Args:
        init_fn (Callable[[PRNGKey, PyTree], SamplerState]): function to initialise the state of chain
        kernel (Callable[[int, PRNGKey, SamplerState], SamplerState]): transition kernel
        get_params (Callable[[SamplerState], PyTree]): functions that gets the target parameters from the state
        pbar (bool, optional): whether or not to display the progress bar. Defaults to True.

    Returns:
        Callable: sampling function with the same signature as kernel
    """

    def sampler(key, Nsamples, state):
        samples = []
        key, subkey = random.split(key)
        #state = init_fn(subkey, params)

        _tqdm = tqdm(range(Nsamples)) if pbar else range(Nsamples)

        for i in _tqdm:
            key, subkey = random.split(key)
            state = kernel(i, subkey, state)
            samples.append(get_params(state))
            #jax.clear_caches()

        samples = tree_stack(samples)

        return samples, state

    return sampler


def zscore_normalization(X, mean=None, std=None, eps=1e-10):
    """Apply z-score normalization on a given data.

    Args:
        X: numpy array, shape [batchsize, num_dims], the input dataset.
        mean: numpy array, shape [num_dims], the given mean of the dataset.
        var: numpy array, shape [num_dims], the given variance of the dataset.

    Returns:
        tuple: the normalized dataset and the resulting mean and variance.
    """
    if X is None:
        return None, None, None

    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / (std + eps)

    return X_normalized, mean, std


def zscore_unnormalization(X_normalized, mean, std):
    """Unnormalize a given dataset.

    Args:
        X_normalized: numpy array, shape [batchsize, num_dims], the
            dataset needs to be unnormalized.
        mean: numpy array, shape [num_dims], the given mean of the dataset.
        var: numpy array, shape [num_dims], the given variance of the dataset.

    Returns:
        numpy array, shape [batch_size, num_dims] the unnormalized dataset.
    """
    return X_normalized * std + mean


def unnormalize_predictions(pred_mean, pred_var, y_mean, y_std):
    """Unnormalize the regression predictions.

    Args:
        pred_mean: np.array, [n_data, 1], the predictive mean.
        pred_var: np.array, [n_data, 1], the predictive variance.
        y_mean: np.array, [n_data, 1], the mean estimated from training data.
        y_std: np.array, [n_data, 1], the std estimated from training data.        
    """
    pred_mean = zscore_unnormalization(pred_mean, y_mean, y_std)
    pred_var = pred_var * (y_std ** 2)

    return pred_mean, pred_var


def normalize_data(X_train, y_train, X_test=None, y_test=None):
    """Wrapper function used to normalize regression datasets.

    Args:
        X_train: np.array, [n_data, n_dims], the inputs of training data.
        y_train: np.array, [n_data, 1], the targets of training data.
        X_test: np.array, [n_data, n_dims], the inputs of test data.
        y_test: np.array, [n_data, 1], the targets of test data.
    """
    # Normalize the dataset
    X_train_, X_mean, X_std = zscore_normalization(X_train)
    y_train_, y_mean, y_std = zscore_normalization(y_train,eps=0.)

    if (X_test is not None) and (y_test is not None):
        X_test_, _, _ = zscore_normalization(X_test, X_mean, X_std)
        y_test_, _, _ = zscore_normalization(y_test, eps=0.)
        return X_train_, y_train_, X_test_, y_test_, y_mean, y_std
    else:
        return X_train_, y_train_, y_mean, y_std

def get_random_x(n_data, X, x_min, x_max, key, real_ratio):
    
    key1, key2 = jax.random.split(key)


    n_real = int(n_data * real_ratio)
    # assert n_real < self.X.shape[0]
    n_real = min(n_real, int(X.shape[0]))
    n_rand = n_data - n_real

    # Choose randomly training inputs
    indices = jax.random.permutation(key, X.shape[0], axis=0, independent=False)[:n_real]

    # Generate random points
    X_real = X[indices, ...]
    X_rand = jax.random.uniform(key1, shape=[n_rand,x_min.shape[0]], minval=x_min, maxval=x_max)  

    # Concatenate both sets
    X = jnp.concatenate((X_real, X_rand), axis=0)
    indices = jax.random.permutation(key2, X.shape[0], axis=0, independent=False)
    X = X[indices, ...]

    return X


def get_input_range(X_train, X_test, ratio=0.0):
    """Get the range of each coordinate of the input data."""
    x_min = np.minimum(X_train.min(axis=0), X_test.min(axis=0))
    x_max = np.maximum(X_train.max(axis=0), X_test.max(axis=0))
    d = x_max - x_min
    x_min = x_min - d * ratio
    x_max = x_max + d * ratio

    return x_min, x_max



def spliterate(tree, chunk_size, n):
    for start in range(0, n, chunk_size):
        yield jax.tree_util.tree_map(lambda t: t[start:start + chunk_size], tree)
  

class Dataset(object):
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None



from retry.api import retry_call
def get_data(dataset, split_num, base):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """

    try:
        from azureml.fsspec import AzureMachineLearningFileSystem

        #from azure.ai.ml import MLClient
        #from azure.identity import DefaultAzureCredential
        #ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        #data_asset = ml_client.data.get("UCI_datasets", version=1)
        #fs = AzureMachineLearningFileSystem(data_asset.path)
        fs = AzureMachineLearningFileSystem(base)


        print("Using Azure Filesystem")
#        open_file = lambda f: retry_call(fs.open, fargs=[f], delay=2, jitter=(1,20))
        open_file = fs.open
    except:
#        open_file = lambda f: retry_call(open, fargs=[f], delay=2, jitter=(1,20))
        open_file = open

    _DATA_DIRECTORY_PATH = base + dataset + "/data/"
    _DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
    _INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
    _INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
    _N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"

    index_train_path = _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    index_test_path = _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt" 


    # We load the data

    with open_file(_DATA_FILE) as f:
        data = np.loadtxt(f)

    # We load the indexes for the features and for the target

    with open_file(_INDEX_FEATURES_FILE) as f:
        index_features = np.loadtxt(f)
    
    with open_file(_INDEX_TARGET_FILE) as f:
        index_target = np.loadtxt(f)

    X = data[ : , [int(i) for i in index_features.tolist()] ]
    y = data[ : , int(index_target.tolist()) ]

    print("data shape: ", X.shape )

    # We load the indexes of the training and test sets
    with open_file(index_train_path) as f:
        index_train = np.loadtxt(f)
    with open_file(index_test_path) as f:
        index_test = np.loadtxt(f)

    X_train = X[ [int(i) for i in index_train.tolist()] ]
    y_train = y[ [int(i) for i in index_train.tolist()] ]
    
    X_test = X[ [int(i) for i in index_test.tolist()] ]
    y_test = y[ [int(i) for i in index_test.tolist()] ]

    data = Dataset
    data.X_train = X_train
    data.Y_train = np.expand_dims(y_train, axis=1)
    data.X_test = X_test
    data.Y_test = np.expand_dims(y_test, axis=1)

    return data




def load_data(dataset, split, data_folder, n, normalize_x):
    y_range = None
    if dataset=="dgp":
        from PIL import Image 

        #image = Image.open("dgp3.png").convert('L')
        #image = Image.open("dgp10.png").convert('L')
        imgpth = os.path.join(os.path.dirname(__file__),"beta.png")
        image = Image.open(imgpth).convert('L')

        #prop = 0.15
        #image = image.resize( [int(prop * s) for s in image.size] )

        pixels = jnp.asarray(image)#, dtype=types.FLOAT_TYPE)

        # DGP IMAGE: extend for plotting 
        #MARGINX = 200
        #MARGINY = 100
        MARGINX = 0
        MARGINY = 0



        # Find coordinates of all pixels below threshold
        threshold_level = 0
        coords = jnp.column_stack(jnp.where(pixels > threshold_level))

        #threshold_level = 254
        #coords = jnp.column_stack(jnp.where(pixels < threshold_level))

        


        pixels_norm = pixels #(255 - pixels)/255
        leftright_margin = jnp.zeros((pixels_norm.shape[0],MARGINX))
        pixels_w_margin = jnp.concatenate([leftright_margin,pixels_norm,leftright_margin], axis=1)
        topbottom_margin = jnp.zeros((MARGINY,pixels_w_margin.shape[1]))
        pixels_w_margin = jnp.concatenate([topbottom_margin,pixels_w_margin,topbottom_margin], axis=0)

        pixels_w_margin = 1.*(pixels_w_margin > threshold_level)
        fname="dgp_true"
        plt.matshow(pixels_w_margin, cmap='PuBu_r')
        savefigs(fname)
        #plt.close()
        #mlflow.log_artifact(fname)
        #os.remove(fname) #don't leave garbage


        data = Dataset

        # inputs are horizontal coordinates
        data_x = coords[:,1].reshape(-1, 1)
        data_y = coords[:,0].reshape(-1, 1)

        data_y = 8*(data_y/pixels.shape[0] - .5)
        data_x = 8*(data_x/pixels.shape[1] - .5)
        #data_x = 8*(data_x/pixels.shape[0] - .5)

        data_y = - data_y #flip

        random_batch = np.random.choice(data_x.shape[0], int(n))
        data.X_train = data_x[random_batch] 
        data.Y_train = data_y[random_batch] 


        random_batch = np.random.choice(data_x.shape[0], int(n))
        data.X_test = data_x[random_batch] 
        data.Y_test = data_y[random_batch] 





        data.X_train, _, data.X_test, _, _, _ = normalize_data(data.X_train, data.Y_train, data.X_test, data.Y_test)

        plt.scatter(data.X_train, data.Y_train, s=1)
        ax = plt.gca()
        ax.set_xlim([-4., 4.])
        ax.set_ylim([-4., 4.])
        ax.set_aspect('equal')
        savefigs("sampled_image")

        #kludg because I am having dimension problem with stax and no time

        data.X_train = np.concatenate([data.X_train,data.X_train], axis=1)
        data.X_test = np.concatenate([data.X_test,data.X_test], axis=1)


        y_range = (-MARGINY,pixels.shape[0]+MARGINY)

    elif dataset!="synth":
        if dataset == "NYTaxiLocationPrediction":
            #from data import NYTaxiLocationPrediction
            from datasets import NCYTaxiDropoffPredict

            nyc = NCYTaxiDropoffPredict(data_folder=data_folder)
            seed=22 # same as in https://github.com/freelunchtheorem/Conditional_Density_Estimation/blob/master/cde/evaluation/empirical_eval/benchmark_empirical.py
            rds = np.random.RandomState(seed)
            ds = zip(*nyc.get_train_valid_splits(valid_portion=0.2, n_splits=3,
                                                   shuffle=True, random_state=rds))

            data = Dataset()
            (X_train, Y_train, X_test, Y_test) = list(ds)[split]    
            data.X_train = X_train
            data.Y_train = Y_train
            data.X_test = X_test
            data.Y_test = Y_test        


        elif False:
            from data import get_regression_data

            ####### LOAD DATA ######
            data = get_regression_data(dataset, split=split)
            # THE SAME TRUNCATION IS PERFORMED in the script for DGP
            data.X_test = data.X_test[:10000]
            data.Y_test = data.Y_test[:10000]
        else:
            ####### LOAD DATA ######
            data = get_data(dataset, split_num=split, base=data_folder)

        if n is not None: # truncate training  and test set
            data.X_train = data.X_train[:int(n)]
            data.Y_train = data.Y_train[:int(n)]

            data.X_test = data.X_test[:int(n)]
            data.Y_test = data.Y_test[:int(n)]

        else:
            n = data.X_train.shape[0]

        #if dataset != "NYTaxiLocationPrediction":
        data.orig_X_train = data.X_train
        data.orig_X_test = data.X_test
        if normalize_x:
            data.X_train, _, data.X_test, _, _, _ = normalize_data(data.X_train, data.Y_train, data.X_test, data.Y_test)
        


    else:
        n = int(n)

        if False:
            import scipy.stats as ss
            # Parameters of the mixture components
            # norm_params = np.array([[5, 1],
            #                         [9, 1.3]])
            norm_params = np.array([[-.3, .1],
                                [1.5, 1.]])

            n_components = norm_params.shape[0]
            # Weight of each component, in this case uniform
            weights = np.ones(n_components, dtype=np.float64) / n_components
            # A stream of indices from which to choose the component
            mixture_idx_train = np.random.choice(len(weights), size=n, replace=True, p=weights)
            
            # y is the mixture sample
            y_train = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx_train),
                            dtype=np.float64)
            mixture_idx_test = np.random.choice(len(weights), size=n, replace=True, p=weights)
            y_test = np.fromiter((ss.norm.rvs(*(norm_params[i])) for i in mixture_idx_test), dtype=np.float64)

            data = Dataset
            data.X_train = np.ones_like(y_train).reshape((-1,1))
            data.X_test = np.ones_like(y_test).reshape((-1,1))
            data.Y_train = y_train.reshape((-1,1))
            data.Y_test = y_test.reshape((-1,1))

            def logdens_gaussmix(y, mus, sigmas, logweights):
                assert len(y.shape)==1 ##one column y element
                logdens = jax.scipy.stats.norm.logpdf(y,mus,sigmas)
                logweightdens = logdens + logweights

                ll = jax.scipy.special.logsumexp(logweightdens)

                return ll


            ll = jnp.mean(vmap(logdens_gaussmix,(0,None,None,None))(jnp.expand_dims(y_test,1),norm_params[:,0],norm_params[:,1], jnp.log(weights)))
            print("true loglikelihood of test set", ll)


        else:
            data = Dataset
            np.random.seed(0)
            data.X_train = np.random.uniform(-1,1,n)
            data.X_test = np.random.uniform(-1,1,n)


            
            
            #s = n//3
            s = n//2
            data.Y_train = get_all(data.X_train, s)
            data.Y_test = get_all(data.X_test, s)
            
            data.X_train = data.X_train.reshape((-1,1))
            data.X_test = data.X_test.reshape((-1,1))
            data.Y_train = data.Y_train.reshape((-1,1))
            data.Y_test = data.Y_test.reshape((-1,1))

            #kludg because I am having dimension problem with stax and no time

            data.X_train = np.concatenate([data.X_train,data.X_train], axis=1)
            data.X_test = np.concatenate([data.X_test,data.X_test], axis=1)

            #data.X_train, _, data.X_test, _, _, _ = normalize_data(data.X_train, data.Y_train, data.X_test, data.Y_test)
            plt.scatter(data.X_train[:,0],data.Y_train)
            plt.axvline(x=0.47976784, color='k', label='axvline - full height')
            savefigs("synth")




            #plt.scatter(data.X_test,data.Y_test)

    return data, y_range


def plot_dgp_posterior(samples, X_test, y_range, chunkyfied_mix_preds):
    # set of values a I want to consider for Y for each X value
    ys = jnp.arange(y_range[0], y_range[1]) ### DUMMY. WE DONT USE IT #, dtype=types.FLOAT_TYPE
    # tile for each X value
    ys = np.tile(ys, (X_test.shape[0],1))
    ys = jnp.expand_dims(ys, axis=2)

    res = vmap(chunkyfied_mix_preds, (None, 1, None)) (samples, X_test, ys )

    fname="dgp_posterior_logcolor"
    plt.matshow(res)
    savefigs(fname) #, bbox_inches='tight')
    #plt.close()
    #mlflow.log_artifact(fname)
    #os.remove(fname) #don't leave garbage

    expres = jnp.exp(res)
    fname="dgp_posterior"
    plt.matshow(expres, cmap='PuBu_r')  #"linear", "log", "symlog", "logit"
    ax = plt.gca()
    savefigs(fname) #), bbox_inches='tight')
    #plt.close()
    #mlflow.log_artifact(fname)
    #os.remove(fname) #don't leave garbage


def plot_image(samples, x_range, y_range, chunkyfied_mix_preds, prefix):
    # number of pixels on each side
    n_pixels = 200
    
    # set of values a I want to consider for Y for each X value
    ys = jnp.linspace(y_range[0], y_range[1], n_pixels) 
    # tile for each X value
    ys = jnp.tile(ys, (n_pixels,1))
    ys = jnp.expand_dims(ys, axis=2)

    xs = jnp.linspace(x_range[0], x_range[1],n_pixels) 
    # kludge, repeat X value
    xs = jnp.stack([xs,xs], 1)

    if False: #"prior" in prefix:
        x = xs[:,0]
        y = ys[0]
        vy_norm = vmap(jax.scipy.stats.norm.pdf,(0,None,None))
        #ground_truth = (vy_norm(y , get_loc_1(x), get_scale_1(x)) + vy_norm(y , get_loc_2(x), get_scale_2(x)) + vy_norm(y , get_loc_3(x), get_scale_3(x)))/3.
        ground_truth = (vy_norm(y , get_loc_1(x), get_scale_1(x)) + vy_norm(y , get_loc_2(x), get_scale_2(x)) )/2.
        fname="image_groundtruth"
        plt.matshow(ground_truth, cmap='PuBu_r', extent=[x_range[0],x_range[1],y_range[0],y_range[1]],aspect="auto", origin='lower')
        savefigs(fname) #, bbox_inches='tight')

    res = vmap(chunkyfied_mix_preds, (None, None, 1)) (samples, xs, ys )[0]

    if False:
        fname=prefix + "_image_logcolor"
        plt.matshow(res, extent=[x_range[0],x_range[1],y_range[0],y_range[1]],aspect="auto", origin='lower')
        savefigs(fname) #, bbox_inches='tight')
        #plt.close()
        #mlflow.log_artifact(fname)
        #os.remove(fname) #don't leave garbage

    expres = jnp.exp(res)
    fname=prefix + "_image"
    plt.matshow(expres, cmap='PuBu_r', extent=[x_range[0],x_range[1],y_range[0],y_range[1]],aspect="auto",origin='lower')  #"linear", "log", "symlog", "logit"
    ax = plt.gca()
    savefigs(fname) #), bbox_inches='tight')
    #plt.close()
    #mlflow.log_artifact(fname)
    #os.remove(fname) #don't leave garbage



from jax import vmap, tree_map
import os


from model import *

# log_normalizer expressed in terms of standard parameters alpha and beta
def log_normalizer(a,b):
    #jç
    #jax.debug.print("🤯 {b} 🤯", b=b)

    return -jax.scipy.special.gammaln(a+b) + jax.scipy.special.gammaln(a) + jax.scipy.special.gammaln(b)


def delta_i_j_plus_log_w(log_w_i, a_i, b_i, log_w_j, a_j, b_j):

    assert len(a_i.shape) == 0 
    # log normalizer is wrt to natural parameters eta_1  = alpha - 1, eta_2 = beta - 1
    delta_ij = log_normalizer(a_i + a_j - 1 , b_i + b_j - 1) - (log_normalizer(a_i,b_i) + log_normalizer(a_j,b_j))

    return log_w_i + log_w_j + delta_ij


v_delta_plus_log_w = vmap(vmap(delta_i_j_plus_log_w,in_axes=(0,0,0,None,None,None)), in_axes=(None,None,None,0,0,0))


#Renyi quadratic entropy
# for one beta mixture, one observation
def h2(dist):
    assert len(dist.logweights.shape) == 1  

    delta_plus_log_w_mat = v_delta_plus_log_w(dist.logweights, dist.alphas, dist.betas, dist.logweights, dist.alphas, dist.betas) 

    return - jax.scipy.special.logsumexp(delta_plus_log_w_mat) ## sum over everything  # should skip weight 0 components

# apply on all dists
v_h2 = vmap(h2, in_axes = 0)

def jensen_renyi(mixture,dists):
    h2_dists = v_h2(dists)
    h2_mixture =  h2(mixture) 
    return h2_mixture - jnp.mean(h2_dists)

#for just two distributions, same number of components
def JR(dist1,dist2):
    dists = jnp.stack([dist1,dist2])
    mixture = batch_mix(dists)

    return jensen_renyi(mixture,dists)

#for just two distributions, possibly different number of components
def JR2(dist1,dist2):
    h2_dist1 = h2(dist1)
    h2_dist2 = h2(dist2)
    mixture = BetaMix(logweights=jnp.concatenate([dist1.logweights,dist2.logweights])- jnp.log(2.),
                     alphas=jnp.concatenate([dist1.alphas,dist2.alphas]),
                     betas=jnp.concatenate([dist1.betas,dist2.betas]))
    h2_mixture = h2(mixture)
    return h2_mixture - (h2_dist1 + h2_dist2)/2.


if True:
    # for one beta mixture, one observation
    # log int_ dist1 * dist2
    def log_int_product(dist1,dist2):
        assert len(dist1.logweights.shape) == 1 and len(dist1.alphas.shape) == 1 and len(dist1.betas.shape)
        assert len(dist2.logweights.shape) == 1 and len(dist2.alphas.shape) == 1 and len(dist2.betas.shape)

        delta_plus_log_w_mat = v_delta_plus_log_w(dist1.logweights, dist1.alphas, dist1.betas, dist2.logweights, dist2.alphas, dist2.betas) 

        return jax.scipy.special.logsumexp(delta_plus_log_w_mat, axis=(0,1))


    def cs_div_log(dist1, dist2):
        return - (log_int_product(dist1,dist2) - .5*(log_int_product(dist1,dist1) + log_int_product(dist2,dist2)))

    v_cs_div_log = vmap(vmap(cs_div_log, (0,None)), (None,0))

    def multiple_cs_div_log(dists):
        log_ratios = - v_cs_div_log(dists,dists)


        C = dists.logweights.shape[0]
        k = (C-1)*C/2

        norm_log_ratios = log_ratios - jnp.log(k)
        norm_log_ratios_upper = jnp.triu(norm_log_ratios, 1)
        norm_log_ratios_upper += jnp.tril(jnp.ones_like(norm_log_ratios_upper) * (-jnp.inf) )

        cs =  -jax.scipy.special.logsumexp(norm_log_ratios_upper)
        return cs
        #return 1-jnp.exp(-cs)



#### MY CS div without log
if True:
    def multiple_expcs_div(dists):
        return 1-jnp.exp(-multiple_cs_div_log(dists))  

    def expcs_div(dist1,dist2):
        return 1-jnp.exp(-cs_div_log(dist1,dist2))

else:
    multiple_cs_div = multiple_cs_div_log
    cs_div = cs_div_log

batch_multiple_expcs_div = vmap(multiple_expcs_div, in_axes=0)

def old_plot(dists, logDens, batch_mix, min_y, max_y, maxdens=None): 
    
    #print("PLOTTING...")
    n_rows = 3
    n_cols = 3 # for the refs plots, +1 for posterior
    fig1, axs1 = plt.subplots(n_rows, n_cols, figsize=(33.867,19.05))

    n_plots = n_rows * n_cols

    # we are just interested in n_plots
    dists =  jax.tree_util.tree_map(lambda x: x[:n_plots], dists)

    points_plus_one = 200

    if min_y is None:
        step = 1/points_plus_one
        ys = np.arange(0+step/2, 1.-step/2, step)
    else:
        step = (max_y-min_y)/points_plus_one
        ys = np.arange(min_y+step/2, max_y-step/2, step)



    # logDens is for one beta mixture, one y
    # first map over ys,then over batch
    vbatchYsLogDens = vmap(vmap(logDens, (0,None)),(None,0))

    # then over M, 
    vbatchYsMLogDens = vmap(vbatchYsLogDens,(None,0))

    ps = vbatchYsMLogDens(ys,dists)
    log_ps = ps
    ps = jnp.exp(ps)

    mixture = batch_mix(dists)
    ps_mix = vbatchYsLogDens(ys, mixture)
    ps_mix = jnp.exp(ps_mix)

    try:
        batch_jensen_renyi = vmap(jensen_renyi, in_axes=(0,0))
        jr = batch_jensen_renyi(mixture,dists)
    except:
        exp_cs=None

    try:
        exp_cs = batch_multiple_expcs_div(dists)
    except:
        jr=None


    ###TODO
    if False: 
        #for each x in the batch, for each sampled_weight, sample y values

        rng_key, beta_key, cat_key = jax.random.split(rng_key, 3)

        M_y = 30

        cat_key = jax.random.split(cat_key, M_y)
        # sample from ys from Normal(0,1)
        c = vmap(jax.random.categorical,in_axes= (0,None,None, None))(key=cat_key, logits= dists.logweights, axis = 2, shape=None)
        ys = jax.random.normal(key=norm_key, shape=(x0.shape[0], M,  M_y))

        mus = jnp.take(dists.mus,c)
        sigmas = jnp.take(dists.sigmas,c)

        y_unnormalize = lambda norm_y,mu,sigma: norm_y*sigma + mu
        ys_unnormalize = vmap(y_unnormalize, in_axes=(2,None,None), out_axes=(2))

        ys = ys_unnormalize(ys,mus,sigmas)



    for i in range(n_plots):

        ax1 = axs1[int(np.ceil(i//n_cols)), i%n_cols]

        ps_mix_i = ps_mix[i]
        ax1.fill_between(ys,ps_mix_i) 
        #ax1.set_title("JS uncertainty")
        
        ax1.fill_between(ys,ps_mix_i, color='b')

        js = 0.

        M = ps.shape[1]
        for j in range(M):

            ps_ij = ps[i,j]

            if False:
                log_ps_ij = log_ps[i,j]
                try:
                    # compare each dist to mix, midpoint approximation of the integral 
                    js += 1/M * np.sum(step * ps_ij * (log_ps_ij - np.log(ps_mix_i))) 
                except:
                    pass

            ax1.plot(ys,ps_ij, color='r', alpha=.1)

        if False:
            #ax1.set_title("JR="+(f'{jr[i]:.4f}')+" JS≈"+(f'{js:.4f}')+" 1-exp(-cs)≈"+(f'{exp_cs[i]:.4f}'))
            try:
                ax1.set_title(" JSD ≈ "+(f'{js:.4f}'))
                #ax1.set_title("JR="+(f'{jr[i]:.4f}')+" 1-exp(-cs)≈"+(f'{exp_cs[i]:.4f}'))
            except:
                pass

        if maxdens is not None:
            ax1.set_ylim(top=maxdens, bottom=0.)


import geopandas
import geoplot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap

# # get colormap
# ncolors = 256
# color_array = plt.get_cmap('gist_rainbow')(range(ncolors))

# # change alpha values
# color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

# # create a colormap object
# map_object = LinearSegmentedColormap.from_list(name='rainbow_alpha',colors=color_array)

# # register this new colormap with matplotlib
# plt.colormaps.register(cmap=map_object)

def plot2D(fname, dists, logDens, batch_mix, min_y, max_y, x=None,  cdf_squeeze = None, logpdf_unsqueeze = None, maxdens=None): 
    
    #print("PLOTTING...")
    if x is None:
        n_rows = 1
        n_cols = 1 # for the refs plots, +1 for posterior
    else:
        n_rows = 3
        n_cols = 3 # for the refs plots, +1 for posterior

    n_plots = n_rows * n_cols
    #fig1, axs1 = plt.subplots(n_rows, n_cols) #, figsize=(15,15))


    # we are just interested in n_plots
    dists =  jax.tree_util.tree_map(lambda x: x[:n_plots], dists)

    points_plus_one = 100
    if min_y is None:
        step = 1/points_plus_one
        ys1 = np.arange(0+step/2, 1.-step/2, step)
        ys2 = ys1
        Y1, Y2 = np.meshgrid(ys1, ys2)
    else:
        ys1 = np.linspace(start=min_y[0], stop=max_y[0], num=points_plus_one)
        ys2 = np.linspace(start=min_y[1], stop=max_y[1], num=points_plus_one)
        Y1, Y2 = np.meshgrid(ys1, ys2)

    YY = np.array([Y1.ravel(), Y2.ravel()]).T

    if cdf_squeeze is not None:
        orig_YY = YY
        YY = vmap(cdf_squeeze,0)(YY)


    # logDens is for one beta mixture, one y
    # first map over ys,then over batch
    vbatchYsLogDens = vmap(vmap(logDens, (0,None)),(None,0))

    vbatchYsLogDens = jit(vbatchYsLogDens)
    # then over M, 
    #vbatchYsMLogDens = vmap(vbatchYsLogDens,(None,0))

    #ps = vbatchYsMLogDens(YY,dists)
    #ps = jnp.exp(ps)

    boroughs = geopandas.read_file(geoplot.datasets.get_path('nyc_boroughs'))


    for i in range(n_plots):

        fig1, ax1 = plt.subplots(1, 1) 

        dist_i = jax.tree_util.tree_map(lambda e: e[i:(i+1)], dists)

        mixture = batch_mix(dist_i)

        def chunkyfied_vbatchYsLogDens(YY, mixture, CHUNK_SIZE =  100):
            all_ps = []
            for chunk in tqdm(spliterate(YY, CHUNK_SIZE, YY.shape[0])):
                ps_mix = vbatchYsLogDens(chunk, mixture)
                all_ps.append(ps_mix)
            
            return jnp.concatenate(all_ps, axis=1)

        if logpdf_unsqueeze is not None:
            ps_mix = jnp.squeeze(chunkyfied_vbatchYsLogDens(YY, mixture)) + vmap(logpdf_unsqueeze)(orig_YY)
        else:
            ps_mix = jnp.squeeze(chunkyfied_vbatchYsLogDens(YY, mixture))

        ps_mix = jnp.exp(ps_mix)

        # if n_plots>1:
        #     ax1 = axs1[int(np.ceil(i//n_cols)), i%n_cols]
        # else:
        #     ax1 = axs1

        if min_y is not None:
            boroughs.plot(ax=ax1,alpha=1.,color="white", edgecolor='k' )

        Z = ps_mix.reshape((len(ys1),len(ys2)))
        #ax1.contour(Y1, Y2, Z)

        # Choose colormap
        cmap = plt.cm.jet

        # Get the colormap colors
        my_cmap = cmap(np.arange(cmap.N))

        # Set alpha
        #my_cmap[:,-1] = np.linspace(0, 1, cmap.N)
        my_cmap[:,-1] = 0.75 # this is to make it appear on the color bar

        # Create new colormap
        my_cmap = ListedColormap(my_cmap)

        pc = ax1.pcolormesh(Y1,Y2,Z, cmap=my_cmap, rasterized=True) #, cmap='YlOrRd')

        ## assume first two coordinates of x are pick-up coordinates
        if x is not None:
            ax1.scatter(x[i,0], x[i,1], marker="x", c='g')
        ax1.set_aspect('equal', 'box')

        if min_y is not None:
            ax1.set_xlim(right=max_y[0], left=min_y[0])
            ax1.set_ylim(top=max_y[1], bottom=min_y[1])

        #fig1.colorbar(pc, ax=ax1)
        ax1.get_xaxis().set_ticks([])
        ax1.get_yaxis().set_ticks([])
        
        savefigs(str(i)+"_"+fname)

        

def plot(dists, logDens, batch_mix, minval, maxval, cdf_squeeze, logpdf_unsqueeze, orig_ys, r_factor, maxdens = None, logpdf_uncond=None): 
    
    if logpdf_uncond:
        n_rows = 1
        n_cols = 1 # for the refs plots, +1 for posterior
        fig1, axs1 = plt.subplots(n_rows, n_cols)
    else:
        n_rows = 3
        n_cols = 3 # for the refs plots, +1 for posterior
        fig1, axs1 = plt.subplots(n_rows, n_cols, figsize=(33.867,19.05))


    n_plots = n_rows * n_cols

    # we are just interested in n_plots
    dists =  jax.tree_util.tree_map(lambda x: x[:n_plots], dists)


    #minval = -3.
    #maxval = 3.
    if logpdf_uncond:
        points_plus_one = 1001
        min_val = np.min(orig_ys) 
        max_val = np.max(orig_ys)
        extension = (max_val - min_val)*.3
        max_val += extension
        min_val -= extension
        step = (max_val-min_val)/points_plus_one
        ys = np.arange(min_val, max_val+step, step)
    else: ## we want the original ys with some increased resolution
        orig_ys = jax.lax.sort(jnp.squeeze(orig_ys))
        all_ys = [orig_ys]
        steps = orig_ys[1:] - orig_ys[:-1]
        r_factor = 3 #increase resolutiom
        for i in range(1,r_factor):
            all_ys.append(orig_ys[:-1] + i*steps/r_factor)

        ys = jax.lax.sort(jnp.concatenate(all_ys))





    # logDens is for one beta mixture, one y
    # first map over ys,then over batch
    vbatchYsLogDens = vmap(vmap(logDens, (0,None)),(None,0))

    # then over M, 
    vbatchYsMLogDens = vmap(vbatchYsLogDens,(None,0))

    #ys_prime = cdf_squeeze(jnp.expand_dims(ys,1)) ,coord=0)
    ys_prime = vmap(cdf_squeeze)(jnp.expand_dims(ys,1))

    #ps = vbatchYsMLogDens(ys_prime,dists) + vmap(logpdf_unsqueeze, (0,None))(jnp.expand_dims(ys,1),0)
    ps = vbatchYsMLogDens(ys_prime,dists) + vmap(logpdf_unsqueeze)(jnp.expand_dims(ys,1))
    log_ps = ps
    ps = jnp.exp(ps)

    
    mixture = batch_mix(dists)
    #ps_mix = vbatchYsLogDens(ys_prime, mixture) + vmap(logpdf_unsqueeze, (0,None))(jnp.expand_dims(ys,1),0)
    ps_mix = vbatchYsLogDens(ys_prime, mixture) + vmap(logpdf_unsqueeze)(jnp.expand_dims(ys,1))
    ps_mix = jnp.exp(ps_mix)

    try:
        batch_jensen_renyi = vmap(jensen_renyi, in_axes=(0,0))
        jr = batch_jensen_renyi(mixture,dists)
    except:
        exp_cs=None

    try:
        exp_cs = batch_multiple_expcs_div(dists)
    except:
        jr=None


    COMPUTE_JS = False

    for i in range(n_plots):

        if n_plots>1:
            ax1 = axs1[int(np.ceil(i//n_cols)), i%n_cols]
        else:
            ax1 = axs1

        ps_mix_i = ps_mix[i]
        ax1.fill_between(ys,ps_mix_i)
        #ax1.set_title("JS uncertainty")
        
        ax1.fill_between(ys,ps_mix_i, color='b')

        js = 0.

        M = ps.shape[1]
        for j in range(M):

            ps_ij = ps[i,j]
            
            if COMPUTE_JS:
                #compare each dist to mix, midpoint approximation of the integral , it would be better to use Nielsen's formula
                log_ps_ij = log_ps[i,j]
                js += 1/M * np.sum(step * ps_ij * (log_ps_ij - np.log(ps_mix_i))) 

            if not logpdf_uncond or np.random.choice(300)==0:
                ax1.plot(ys,ps_ij, color='r', alpha=.1)

            if COMPUTE_JS:
                try:
                    #ax1.set_title("JR="+(f'{jr[i]:.4f}')+" 1-exp(-cs)≈"+(f'{exp_cs[i]:.4f}'))
                    ax1.set_title(" JS≈"+(f'{js:.4f}'))
                except:
                    pass

        if logpdf_uncond is not None:
            ps_uncond = np.exp(logpdf_uncond(np.expand_dims(ys,1)))

            ax1.plot(ys,ps_uncond, color='g', alpha=.9, linewidth=3.)
            ax1.set_ylim(top=np.max(ps_uncond)*1.1, bottom=0.)    

        if maxdens is not None:
            ax1.set_ylim(top=maxdens, bottom=0.)


def plot_x(dists, x, logDens, batch_mix, minval, maxval, cdf_squeeze, logpdf_unsqueeze, orig_ys, r_factor, maxdens = None): 
    
    print("PLOTTING...x:")

    if True:
        print(x[0])
        n_rows = 1
        n_cols = 1 # for the refs plots, +1 for posterior
        fig1, axs1 = plt.subplots(n_rows, n_cols)
    else:
        print(x)
        n_rows = 3
        n_cols = 3 # for the refs plots, +1 for posterior
        fig1, axs1 = plt.subplots(n_rows, n_cols, figsize=(33.867,19.05))

    n_plots = n_rows * n_cols

    # we are just interested in n_plots
    dists =  jax.tree_util.tree_map(lambda x: x[:n_plots], dists)


    #minval = -3.
    #maxval = 3.
    if False:
        step = 0.001 
        #(maxval-minval)/points_plus_one
        ys = np.arange(minval, maxval+step, step)
    else: ## we want the original ys with some increased resolution
        orig_ys = jax.lax.sort(jnp.squeeze(orig_ys))
        all_ys = [orig_ys]
        steps = orig_ys[1:] - orig_ys[:-1]
        r_factor = 3 #increase resolutiom
        for i in range(1,r_factor):
            all_ys.append(orig_ys[:-1] + i*steps/r_factor)

        ys = jax.lax.sort(jnp.concatenate(all_ys))



    # logDens is for one beta mixture, one y
    # first map over ys,then over batch
    vbatchYsLogDens = vmap(vmap(logDens, (0,None)),(None,0))

    # then over M, 
    vbatchYsMLogDens = vmap(vbatchYsLogDens,(None,0))

    #ys_prime = cdf_squeeze(jnp.expand_dims(ys,1)) ,coord=0)
    ys_prime = vmap(cdf_squeeze)(jnp.expand_dims(ys,1))

    #ps = vbatchYsMLogDens(ys_prime,dists) + vmap(logpdf_unsqueeze, (0,None))(jnp.expand_dims(ys,1),0)
    ps = vbatchYsMLogDens(ys_prime,dists) + vmap(logpdf_unsqueeze)(jnp.expand_dims(ys,1))
    log_ps = ps
    ps = jnp.exp(ps)

    
    mixture = batch_mix(dists)
    #ps_mix = vbatchYsLogDens(ys_prime, mixture) + vmap(logpdf_unsqueeze, (0,None))(jnp.expand_dims(ys,1),0)
    ps_mix = vbatchYsLogDens(ys_prime, mixture) + vmap(logpdf_unsqueeze)(jnp.expand_dims(ys,1))
    ps_mix = jnp.exp(ps_mix)

    try:
        batch_jensen_renyi = vmap(jensen_renyi, in_axes=(0,0))
        jr = batch_jensen_renyi(mixture,dists)
    except:
        exp_cs=None

    try:
        exp_cs = batch_multiple_expcs_div(dists)
    except:
        jr=None




    for i in range(n_plots):

        if n_plots>1:
            ax1 = axs1[int(np.ceil(i//n_cols)), i%n_cols]
        else:
            ax1 = axs1

        ps_mix_i = ps_mix[i]
        ax1.fill_between(ys,ps_mix_i)
        #ax1.set_title("JS uncertainty")
        
        ax1.fill_between(ys,ps_mix_i, color='b')

        js = 0.

        M = ps.shape[1]
        for j in range(M):

            ps_ij = ps[i,j]
            #log_ps_ij = log_ps[i,j]

            # compare each dist to mix, midpoint approximation of the integral 
            #js += 1/M * np.sum(steps * ps_ij * (log_ps_ij - np.log(ps_mix_i))) 

            ax1.plot(ys,ps_ij, color='r', alpha=.1)

            ps_truth = get_ps_truth(x[i,0],ys) # x[i,0] becaus of kludge of repeated x

            ax1.plot(ys,ps_truth, color='k')


        #try:
            #ax1.set_title("JR="+(f'{jr[i]:.4f}')+" 1-exp(-cs)≈"+(f'{exp_cs[i]:.4f}'))
            #ax1.set_title(" JS≈"+(f'{js:.4f}'))
        #except:
        #    pass

        if maxdens is not None:
            ax1.set_ylim(top=maxdens, bottom=0.)



def plot_save_mlflow(fname):
    savefigs(fname) #, bbox_inches='tight')
    #plt.close()
    #mlflow.log_artifact(fname)
    #os.remove(fname) #don't leave garbage

def old_plot_ml_flow(dists,fname, min_y, max_y, logDens, batch_mix, maxdens=None):
    old_plot(dists, logDens, batch_mix, min_y, max_y, maxdens)
    savefigs(fname) #, bbox_inches='tight')
    #plt.close()
    #mlflow.log_artifact(fname)
    #os.remove(fname) #don't leave garbage

def plot2D_ml_flow(dists,fname, min_y, max_y, logDens, batch_mix, x=None, cdf_squeeze = None, logpdf_unsqueeze = None, maxdens=None):
    plot2D(fname,dists, logDens, batch_mix, min_y, max_y, x, cdf_squeeze , logpdf_unsqueeze, maxdens )
    #savefigs(fname) #, bbox_inches='tight')



def plot_ml_flow(dists,fname, logDens, batch_mix, minval, maxval, cdf_squeeze, logpdf_unsqueeze, orig_ys, r_factor, maxdens = None, logpdf_uncond=None):
    plot(dists, logDens, batch_mix, minval, maxval, cdf_squeeze, logpdf_unsqueeze, orig_ys, r_factor, maxdens, logpdf_uncond)
    savefigs(fname) #, bbox_inches='tight')
    #plt.close()
    #mlflow.log_artifact(fname)
    #os.remove(fname) #don't leave garbage


def plot_ml_flow_x(dists,fname, x ,logDens, batch_mix, minval, maxval, cdf_squeeze, logpdf_unsqueeze, orig_ys, r_factor, maxdens = None):
    plot_x(dists, x, logDens, batch_mix, minval, maxval, cdf_squeeze, logpdf_unsqueeze, orig_ys, r_factor, maxdens)
    savefigs(fname) #, bbox_inches='tight')
    #plt.close()
    #mlflow.log_artifact(fname)
    #os.remove(fname) #don't leave garbage


def plot_learning_curve(X_train, Y_train, batch_loglikelihood, num_chains, samples):
    SUB_SIZE = 1000
    sub_indices  = np.random.choice(X_train.shape[0], size=np.minimum(SUB_SIZE,X_train.shape[0]), replace=False)
    sub_x_train = X_train[sub_indices]
    sub_y_train_norm = Y_train[sub_indices]
    sub_y_train = Y_train[sub_indices]

    #make it lighter for plotting
    light_samples = [None] * num_chains
    for chain in range(num_chains):
        # remove 10%
        light_samples[chain] = tree_map(lambda x: x[::10], samples[chain])
    print("plot mcmc curve")
    max_so_far = - jnp.inf
    for sample in light_samples: #for each chain
        all_ll = []
        CHUNK_SIZE =  1000
        for chunk in tqdm(spliterate(sample, CHUNK_SIZE, len(sample[0][0]))):
            ll = vmap(batch_loglikelihood,(0,None,None))(chunk, sub_x_train, sub_y_train_norm)
            #ll += jnp.squeeze(vmap(logpdf_unsqueeze)(sub_y_train)) 
            ll = jnp.mean(ll,axis=1) # mean over the batch
            all_ll.append(ll)
        
        all_ll = jnp.concatenate(all_ll)
        max_so_far =  jnp.maximum(max_so_far,jnp.max(all_ll))

        # PLOT
        plt.plot(all_ll)
    fname = "full_mcmc_subtrain" 
    plt.ylim(jnp.percentile(all_ll,.8), max_so_far)
    savefigs(fname)
    #plt.close()
    #mlflow.log_artifact(fname)
    #os.remove(fname) #don't leave garbage


def plot_one_learning_curve(key, X_train, Y_train, batch_loglikelihood, samples):
    SUB_SIZE = 200
    sub_indices  = jax.random.choice(key, X_train.shape[0], shape=(np.minimum(SUB_SIZE,X_train.shape[0]),), replace=False)
    sub_x_train = X_train[sub_indices]
    sub_y_train = Y_train[sub_indices]
    #sub_x_train = X_train[:SUB_SIZE]
    #sub_y_train = Y_train[:SUB_SIZE]

    #sub_x_train = X_train
    #sub_y_train = Y_train
    

    #make it lighter for plotting
    light_sample = tree_map(lambda x: x[::10], samples)
    #light_sample = samples
    print("plot one mcmc curve")

    all_ll = []
    CHUNK_SIZE =  20
    for chunk in tqdm(spliterate(light_sample, CHUNK_SIZE, len(light_sample[0][0]))):
        ll = vmap(batch_loglikelihood,(0,None,None))(chunk, sub_x_train, sub_y_train)
        #ll += jnp.squeeze(vmap(logpdf_unsqueeze)(sub_y_train)) 
        ll = jnp.mean(ll,axis=1) # mean over the batch
        all_ll.append(ll)
    
    all_ll = jnp.concatenate(all_ll)

    # PLOT
    plt.plot(all_ll)

    return jnp.percentile(all_ll,.9), jnp.max(all_ll)




# params is needed to the have the tree structure and the parameters of the normal distribution 
def get_random_position(rng_key, params):
    rng_key, init_key = jax.random.split(rng_key)
    initial_position =  tree_random_normal(init_key, params, 1)
    return jax.tree_util.tree_map(lambda x: jnp.squeeze(x, axis=0),initial_position)


from util import *
from einops import rearrange


import blackjax
import optax

def inference_loop(rng_key, step_fn, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = step_fn(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

# online_prior_params: are the parameters of the Gaussian priors on each parameter, The initial position is sampled from it
def fit(
    rng_key,
    sample_prior_params,
    online_prior_params,
    logprob,
    num_warmup=1000,
    num_samples=500,
    hmc=False,
    **params
):
    (
        init_key,
        warmup_key,
        inference_key,
    ) = jax.random.split(rng_key, 3)

    if hmc:
        kernel_type = blackjax.hmc
    else:
        kernel_type = blackjax.nuts

    if online_prior_params is not None:
        initial_position = get_random_position(init_key, online_prior_params)
    else:
        initial_position = sample_prior_params(init_key,1)


    # warm up
    print("warming up")
    adapt = blackjax.window_adaptation(kernel_type, logprob, progress_bar=True,   **params)
    (final_state, params), _ = adapt.run(warmup_key, initial_position, num_warmup)
    
    step_fn = kernel_type(logprob, **params).step
    print(params)
    # inference
    print("sampling")
    states = inference_loop(inference_key, step_fn, final_state, num_samples)
    samples = states.position

    return samples


def batch_data(rng_key, data, batch_size, data_size):
    """Return an iterator over batches of data."""
    while True:
        _, rng_key = jax.random.split(rng_key)
        idx = jax.random.choice(
            key=rng_key, a=jnp.arange(data_size), shape=(batch_size,)
        )
        minibatch = tuple(elem[idx] for elem in data)
        yield minibatch



# online_prior_params: are the parameters of the Gaussian priors on each parameter, The initial position is sampled from it
def fit_sg(
    rng_key,
    sample_prior_params,
    online_prior_params,
    logprior,
    loglik,
    data_size,
    batches,
    step_size,
    num_warmup,
    num_samples=500,
    hmc=False,
    maxnorm=np.inf,
    **params
):
    (
        init_key,
        warmup_key,
        inference_key,
    ) = jax.random.split(rng_key, 3)


    from blackjax.sgmcmc.gradients import grad_estimator

    grad_fn_ = grad_estimator(logprior, loglik, data_size)

    if not np.isinf(maxnorm):
        p_clip_grads = partial(clip_grads, max_norm = maxnorm)
        grad_fn = lambda p, data: p_clip_grads(grad_fn_(p, data)) # p is the current position (parameter space)
    else:
        grad_fn = grad_fn_

    if hmc:
        sghmc = blackjax.sghmc(grad_fn, **params)
        step_fn = jax.jit(sghmc.step)
    else:
        sgld = blackjax.sgld(grad_fn)
        step_fn = jax.jit(sgld.step)
        #step_fn = sgld.step


    if online_prior_params is not None:
        position = get_random_position(init_key, online_prior_params)
    else:
        position = sample_prior_params(init_key,1)

    # warm up  ## DISCARD WARMUP SAMPLES TO AVOID EXCESSIVE MEMORY CONSUMPTION
    print("warming up")
    for i in tqdm(range(num_warmup), mininterval=5):
        minibatch = next(batches)
        rng_key, sample_key = jax.random.split(rng_key)
        position = step_fn(sample_key, position, minibatch, step_size)


    samples = []
    print("sampling")
    for i in tqdm(range(num_samples), mininterval=5):
        minibatch = next(batches)
        rng_key, sample_key = jax.random.split(rng_key)
        position = step_fn(sample_key, position, minibatch, step_size)
        samples.append(position)


    return tree_stack(samples)

# here the params_structure are weight and bias of each layer. I want to approximate the posterior distribution over them
def fit_vi(
    rng_key,
    params_structure,
    logprob,
    num_samples,
    learning_rate,
    iterations,
    vi_samples,
    **params
):


    my_optimiser = optax.flatten(optax.adam(learning_rate))

    state = blackjax.vi.meanfield_vi.init(params_structure, my_optimiser)

    print("optimizing")
    all_samples = []

    for i in tqdm(range(iterations)):
        rng_key, step_key = jax.random.split(rng_key, 2)

        state = blackjax.vi.meanfield_vi.step(step_key, state, logprob, my_optimiser, vi_samples )[0]

        ## sample just for plotting the learning curve
        rng_key, sample_key = jax.random.split(rng_key, 2)
        all_samples.append( blackjax.vi.meanfield_vi.sample(sample_key, state, 1))


    print("sampling")
    rng_key, sample_key = jax.random.split(rng_key, 2)
    all_samples.append( blackjax.vi.meanfield_vi.sample(sample_key, state, num_samples))

    samples = tree_concat(all_samples)

    return samples


# here the params_structure are weight and bias of each layer. I want to approximate the posterior distribution over them
def fit_vi_batched(
    rng_key,
    params_structure,
    logprior,
    loglik,
    batches,
    num_samples,
    learning_rate,
    iterations,
    vi_samples,
    **params
):


    my_optimiser = optax.flatten(optax.adam(learning_rate))

    state = blackjax.vi.meanfield_vi.init(params_structure, my_optimiser)

    print("optimizing")

    all_samples = []
    for i in tqdm(range(iterations)):
        minibatch = next(batches)
        rng_key, step_key = jax.random.split(rng_key)

        loglik = partial(loglik, data= minibatch)

        def joint_logdensity(params): #unnormalized 
            return logprior(params) + loglik(params)

        state = blackjax.vi.meanfield_vi.step(step_key, state, joint_logdensity, my_optimiser, vi_samples )[0]

        ## sample just for plotting the learning curve
        rng_key, sample_key = jax.random.split(rng_key, 2)
        all_samples.append( blackjax.vi.meanfield_vi.sample(sample_key, state, 1))


    print("sampling")
    rng_key, sample_key = jax.random.split(rng_key, 2)
    all_samples.append( blackjax.vi.meanfield_vi.sample(sample_key, state, num_samples))

    samples = tree_concat(all_samples)

    return samples


def sample_posterior_blackjax(validation, rng_key, loglikelihood, logprior, num_chains, num_warmup, num_samples, keep_every, sg, sampler, step_size, vi_samples, batch_size, leap_frog_steps, X_train, Y_train, prior_params, sample_prior_params, params_structure, maxnorm):
    batch_loglikelihood = vmap(loglikelihood, in_axes=(None, 0, 0))
    if not sg: 
        ## use FULL DATASET
        def loglik(params,X , y):
            return jnp.sum(batch_loglikelihood(params, X, y)) 

        loglik = partial(loglik, X= X_train, y=  Y_train)

        def joint_logdensity(params): #unnormalized 
            return logprior(params) + loglik(params)


        if sampler == "nuts": ### NUTS VERSION THAT WORKS, expensive to compute of course
            print("NUTS")
            num_chains = num_chains

            all_samples = []
            for i in range(num_chains):
                rng_key, fit_key = jax.random.split(rng_key,2)

                samples = fit(fit_key, sample_prior_params, prior_params, joint_logdensity, num_warmup=num_warmup, num_samples=num_samples//num_chains,  hmc=False, initial_step_size=step_size) #, num_integration_steps=ARGS.leap_frog_steps)

                all_samples.append(samples)

            samples = tree_stack(all_samples, axis=1)

        elif sampler == "vi": #VI with the whole dataset
            print("VI FULL DATASET")
            rng_key, fit_key = jax.random.split(rng_key,2)
            samples = fit_vi(fit_key, params_structure, joint_logdensity, num_samples=num_samples, learning_rate=step_size, iterations=num_warmup, vi_samples=vi_samples)



    else:
        rng_key, batch_key = jax.random.split(rng_key, 2)

        data_size = X_train.shape[0]

        # Batch the data
        batches = batch_data(batch_key, (X_train, Y_train), batch_size, data_size)


        if sampler == "vi": #VI VERSION
            print("VI BATCHED")


            def loglik(params, data):
                X , y = data
                return jnp.sum(batch_loglikelihood(params, X, y)) 


            rng_key, fit_key = jax.random.split(rng_key,2)
 
            samples = fit_vi_batched(fit_key, params_structure, logprior, loglik, batches, num_samples=num_samples, learning_rate=step_size, iterations=num_warmup, vi_samples=vi_samples)


        else:  
            print("BLACKJAX SGMCMC")

            def loglik(params, data):
                X , y = data
                return jnp.sum(loglikelihood(params, X, y)) 


            all_samples = []
            for i in range(num_chains):
                rng_key, fit_key = jax.random.split(rng_key,2)
                hmc = (leap_frog_steps is not None)
                samples = fit_sg(fit_key, sample_prior_params, prior_params, logprior, loglik, data_size, batches, step_size=step_size, num_warmup = num_warmup, num_samples= keep_every * (num_samples//num_chains),  hmc=hmc, num_integration_steps=leap_frog_steps, maxnorm=maxnorm)

                all_samples.append(samples)

            samples = tree_stack(all_samples, axis=1)

    if sampler == "vi":
        #add chain dimension to make it compatible with the rest
        #we want "m nc"
        num_chains = 1
        samples = tree_map(lambda e: jnp.expand_dims(e,1),samples)



    if not validation: ### removed during rebuttal for faster execution

        #plot learning curve on part of train set
        SUB_SIZE = 1000
        sub_indices  = np.random.choice(X_train.shape[0], size=np.minimum(SUB_SIZE, X_train.shape[0]), replace=False)
        sub_x_train = X_train[sub_indices]
        #sub_y_train_norm = Y_train_squeezed[sub_indices]
        sub_y_train = Y_train[sub_indices]

        #make it lighter for plotting
        #light_samples = [None] * ARGS.num_chains
        #for chain in range(ARGS.num_chains):
        #    light_samples[chain] = jax.tree_util.tree_map(lambda x: x[::10], samples[chain])

        # remove 10%
        light_samples = tree_map(lambda x: x[::10], samples)

        max_val = -jnp.inf
        #min_val = +jnp.inf
        min_percentile = jnp.inf
        print("plot mcmc curve")
        for i in range(num_chains): #for each chain
            sample = jax.tree_util.tree_map(lambda x: x[:,i], light_samples)
            all_ll = []
            CHUNK_SIZE =  50
            for chunk in tqdm(spliterate(sample, CHUNK_SIZE, len(sample[0][0]))):
                ll = vmap(batch_loglikelihood,(0,None,None))(chunk, sub_x_train, sub_y_train)
                ll = jnp.mean(ll,axis=1) # mean over the batch
                all_ll.append(ll)
            
            all_ll = jnp.concatenate(all_ll)

            # PLOT
            plt.plot(all_ll)
            max_val = jnp.maximum(jnp.max(all_ll),max_val)
            #min_val = jnp.minimum(jnp.min(all_ll),min_val)
            min_percentile = jnp.minimum(jnp.percentile(all_ll,.8),min_percentile)

            
        fname = "full_mcmc_subtrain" 
        
        plt.ylim(min_percentile, max_val ) 

        savefigs(fname) #, bbox_inches='tight')
        #plt.close()
        #mlflow.log_artifact(fname)
        #os.remove(fname) #don't leave garbage

    if sampler == "vi": ## remove samples obtained during ELBO training
        samples = tree_map(lambda x: x[num_warmup:], samples)
    elif sg:
        #samples = tree_map(lambda x: x[num_warmup::keep_every], samples)   
        samples = tree_map(lambda x: x[::keep_every], samples)  ### I DISCARDED WARMUP SAMPLES TO AVOID EXCESSIVE MEMORY CONSUMPTION

    return samples


from jax import tree_map, grad, value_and_grad
from sgmcmcjax.kernels import build_sgld_kernel,build_sghmc_kernel, build_badodab_kernel, build_sgldAdam_kernel, build_sghmc_SVRG_kernel, build_sgld_SVRG_kernel, build_psgld_kernel

from sgmcmcjax.kernels import build_gradient_estimation_fn, sgldAdam, _build_langevin_kernel


#Clip the gradient norm of an iterable of parameters.
#The norm is computed over all gradients together, as if they were concatenated into a single vector. Gradients are modified in-place.
def clip_grad_norm(grad, max_norm=200.):
    #norm = jnp.linalg.norm(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.linalg.norm, grad)))
    norm = jnp.linalg.norm(jnp.stack(jax.tree_util.tree_leaves(jax.tree_util.tree_map(jnp.linalg.norm, grad))))
    jax.debug.print("norm {norm}", norm=norm)
    clip = lambda x: jnp.where(norm < max_norm, x, x * max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(clip, grad)

def bad_clip_grad_norm(g, max_norm=200.):
    # Simulates torch.nn.utils.clip_grad_norm_
    g, treedef = jax.tree_flatten(g)
    total_norm = jnp.linalg.norm(jnp.array(map(jnp.linalg.norm, g)))
    clip_coeff = jnp.minimum(max_norm / (total_norm + 1e-6), 1)
    g = [clip_coeff * g_ for g_ in g]
    return treedef.unflatten(g), total_norm

#from jax.example_libraries import optimizers

def l2_norm(tree):
  """Compute the l2 norm of a pytree of arrays. Useful for weight decay."""
  leaves, _ = tree_flatten(tree)
  return jnp.sqrt(sum(jnp.vdot(x, x) for x in leaves))

def clip_grads(grad_tree, max_norm):
  """Clip gradients stored as a pytree of arrays to maximum norm `max_norm`."""
  norm = l2_norm(grad_tree)
  #jax.debug.print("norm {norm}", norm=norm)
  normalize = lambda g: jnp.where(norm < max_norm, g, g * (max_norm / norm))
  return tree_map(normalize, grad_tree)

def my_build_grad_log_post(
    loglikelihood: Callable, logprior: Callable, data: Tuple, with_val: bool = False, maxnorm: float = np.inf
) -> Callable:
    """Build the gradient of the log-posterior.
    The returned function has signature:

    grad_lost_post (Callable)
        Args:
            param (Pytree): parameters to evaluate the log-posterior at
            args: data (either minibatch or fullbatch) to pass in to the log-likelihood
        Returns:
            gradient of the log-posterior (PyTree), and optionally the value of the log-posterior (float)

    Args:
        loglikelihood (Callable): log-likelihood for a single data point
        logprior (Callable): log-prior for a single data point
        data (Tuple): tuple of data. It should either have a single array (for unsupervised problems) or have two arrays (for supervised problems)
        with_val (bool, optional): Whether or not the returned function also inclues the value of the log-posterior as well as the value of the gradient. Defaults to False.

    Raises:
        ValueError: the 'data' argument should either be a tuple of size 1 or 2

    Returns:
        Callable: The gradient of the log-posterior
    """
    if len(data) == 1:
        batch_loglik = jit(vmap(loglikelihood, in_axes=(None, 0)))
    elif len(data) == 2:
        batch_loglik = jit(vmap(loglikelihood, in_axes=(None, 0, 0)))
    else:
        raise ValueError("'data' must be a tuple of size 1 or 2")

    Ndata = data[0].shape[0]

    def log_post(param, *args):
        return logprior(param) + Ndata * jnp.mean(batch_loglik(param, *args), axis=0)

    if with_val: ### NOT USED, OTHERWISE I NEED TO PUT MAXNORM CLIPPING TOO
        if np.isinf(maxnorm):
            grad_log_post = jit(value_and_grad(log_post))
            #grad_log_post = value_and_grad(log_post)
        else:
            exit(1) # not implemented
    else:
        if np.isinf(maxnorm):
            grad_log_post = jit(grad(log_post))
            #grad_log_post = grad(log_post)

        else:
            p_clip_grads =partial(clip_grads, max_norm = maxnorm)
            grad_log_post = jit(lambda p, x, y: p_clip_grads(grad(log_post)(p, x, y)))
    return grad_log_post

def my_build_sgldAdam_kernel(
    dt: float,
    loglikelihood: Callable,
    logprior: Callable,
    data: Tuple,
    batch_size: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    maxnorm: float = np.inf
) -> Tuple[Callable, Callable, Callable]:
    """build SGLD-adam kernel. See appendix in paper: https://arxiv.org/abs/2105.13059v1

    Args:
        dt (float): step size
        loglikelihood (Callable): log-likelihood for a single data point
        logprior (Callable): log-prior for a single data point
        data (Tuple): tuple of data. It should either have a single array (for unsupervised problems) or have two arrays (for supervised problems)
        batch_size (int): batch size
        beta1 (float, optional): weights for the first moment of the gradients. Defaults to 0.9.
        beta2 (float, optional): weights for the second moment of the gradients. Defaults to 0.999.
        eps (float, optional): small value to avoid instabilities. Defaults to 1e-8.

    Returns:
        Tuple[Callable, Callable, Callable]: An (init_fun, kernel, get_params) triple.
    """
    grad_log_post = my_build_grad_log_post(loglikelihood, logprior, data, maxnorm = maxnorm)
    estimate_gradient, init_gradient = build_gradient_estimation_fn(
        grad_log_post, data, batch_size
    )
    init_diff, update_diff, get_p_diff = sgldAdam(dt, beta1, beta2, eps)
    init_fn, sgldAdam_kernel, get_params = _build_langevin_kernel(
        init_diff, update_diff, get_p_diff, estimate_gradient, init_gradient
    )
    return init_fn, sgldAdam_kernel, get_params


def sample_posterior_sgmcmcjax(validation, rng_key, loglikelihood, logprior, num_chains, num_warmup, num_samples, keep_every, step_size, batch_size, leap_frog_steps, X_train, Y_train, prior_params, sample_prior_params, maxnorm, noncompiled):
    batch_loglikelihood = jit(vmap(loglikelihood, in_axes=(None, 0, 0)))

    # The tuning parameters in the update equation are 𝑑𝑡, 𝛼, and 𝛽. 
    # The original paper recommends a small value for 𝛼 (such as 0.01) and 𝛽=0. 
    # The number of leapfrog steps 𝐿 must also be tuned.
    # eta : 0.1, 0.2, 0.4, 0.8
    if leap_frog_steps:
        print("Using build_sghmc_kernel")
        #update_rate = 100
        init_fn, my_kernel, get_params = build_sghmc_kernel(step_size, leap_frog_steps, loglikelihood, logprior, (X_train, Y_train), batch_size) #, alpha=ARGS.friction) ### give nans
        #init_fn, my_kernel, get_params = build_sgld_kernel(step_size, loglikelihood, logprior, (X_train, Y_train), batch_size) #
        #update_rate = 1000
        #init_fn, my_kernel, get_params = build_sghmc_SVRG_kernel(ARGS.step_size, ARGS.leap_frog_steps, loglikelihood, logprior, (data.X_train, Y_train_squeezed), ARGS.batch_size, update_rate) ### give nans

    else:
        print("Using build_sgldAdam_kernel")
        #update_rate = 100
        #init_fn, my_kernel, get_params = build_sgld_SVRG_kernel(step_size, loglikelihood, logprior, (X_train, Y_train), batch_size, update_rate=update_rate) #
        init_fn, my_kernel, get_params = my_build_sgldAdam_kernel(step_size, loglikelihood, logprior, (X_train, Y_train), batch_size, maxnorm = maxnorm) #
        #init_fn, my_kernel, get_params = build_sgldAdam_kernel(step_size, loglikelihood, logprior, (X_train, Y_train), batch_size) #

    if noncompiled:
        my_sampler = _build_noncompiled_sampler(init_fn, my_kernel, get_params)
    else:
        my_sampler = _build_compiled_sampler(init_fn, my_kernel, get_params)
 
    num_chains = num_chains
    keep_every = keep_every

    state = [None] * num_chains
    samples  = [None] * num_chains


    NSamples = num_warmup + (num_samples * keep_every)//num_chains

    # for plotting
    min_percentile_so_far = jnp.inf
    max_val_so_far = -jnp.inf
    print("key before starting sampling: ", rng_key)
    for chain in range(num_chains):
        print("chain: ",chain)
        # Now we need to get initial values for the parameters, and we simply sample from their prior distribution:
        ### sample initial position of the chain from the trained prior, I am using a random  params_IC  just to have the structure of the network
        rng_key, subkey, init_key = jax.random.split(rng_key, 3)

        
        if prior_params is not None:
            params = prior_params
            params_IC =  tree_random_normal(init_key, params, 1)
            params_IC = jax.tree_util.tree_map(lambda param: jnp.squeeze(param, axis=0), params_IC) ##remove batch dim
        else:
            params_IC = sample_prior_params(init_key,1)

        state[chain] = init_fn(subkey, params_IC)

        if False:
            CHUNKSIZE=20000
            n_chunks = NSamples//CHUNKSIZE

            all_samples = []
            for _ in range(n_chunks):
                rng_key, subkey = jax.random.split(rng_key, 2)
                samples, state[chain] = my_sampler(subkey, CHUNKSIZE, state[chain])
                all_samples.append(samples)

            REST = NSamples%CHUNKSIZE
            if REST>0:
                rng_key, subkey = jax.random.split(rng_key, 2)
                samples, state[chain] = my_sampler(subkey, REST, state[chain])
                all_samples.append(samples)

            if len(all_samples)>1:
                samples[chain] = tree_concat(all_samples)
            else:
                samples[chain] = all_samples[0]
        else:
            rng_key, subkey = jax.random.split(rng_key, 2)
            samples[chain], state[chain] = my_sampler(subkey, NSamples, state[chain])

        rng_key, plotkey = jax.random.split(rng_key, 2)
        percentile, maxval = plot_one_learning_curve(plotkey,X_train, Y_train, batch_loglikelihood, samples[chain])

        min_percentile_so_far = jnp.minimum(percentile,min_percentile_so_far)
        max_val_so_far = jnp.maximum(maxval,max_val_so_far)


        # remove warmup, keep every
        # start:stop:step
        samples[chain] = tree_map(lambda x: x[num_warmup::keep_every], samples[chain])

        #print("final  state: ", state[chain])

    fname = "full_mcmc_subtrain" 
    #plt.ylim(4., 5.5)
    plt.ylim(min_percentile_so_far, max_val_so_far)
    savefigs(fname) #, bbox_inches='tight')
    #plt.close()
    #mlflow.log_artifact(fname)
    #os.remove(fname) #don't leave garbage

  
    # plot_learning_curve(data.X_train, data.Y_train, batch_loglikelihood, num_chains, samples)
    # for chain in range(num_chains):
    #     # remove warmup, keep every
    #     # start:stop:step
    #     samples[chain] = tree_map(lambda x: x[num_warmup::keep_every], samples[chain])


    # put chain in dimension 1
    samples = tree_stack(samples, axis=1)
    return samples