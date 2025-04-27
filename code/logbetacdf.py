import jax
from jax import numpy as jnp
from jax.scipy.special import logsumexp

def log_beta_cdf_trap(y, a, b, N=1000):

    xvs = jnp.linspace(0.,y, N)[1:] ## remove first segment


    lys = jax.scipy.stats.beta.logpdf(xvs,a,b)
    #jax.debug.print("y {y} alpha {alpha} beta {beta}  lys {lys} ", y=y, alpha=a, beta=b, lys=lys )


    # perform trapezium rule integration
    logdeltas = jnp.log(jnp.diff(xvs, axis=0))
    logI = -jnp.log(2.) + logsumexp(jnp.array([logsumexp(lys[:-1]+logdeltas), logsumexp(lys[1:]+logdeltas)]))
    return logI


def log_beta_cdf_rect(y, a, b, N=500):

    xvs = jnp.linspace(0.,y, N)

    deltas = xvs[1:]-xvs[:-1]
    midpoints = xvs[:-1] + deltas/2.

    lys = jax.scipy.stats.beta.logpdf(midpoints,a,b)

    logdeltas = jnp.log(deltas)

    # perform rectangle rule integration
    logI = logsumexp(lys+logdeltas)

    return logI

#from tensorflow_probability.substrates import jax as tfp
def log_beta_cdf_trick(y,a,b):
    return jnp.log1p(-tfp.math.betainc(a, b, y)) ## if alphas and betas are big, can give 0 1 values then z is inf and nans later 





