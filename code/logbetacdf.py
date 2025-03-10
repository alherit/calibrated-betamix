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


if False:


    def log_beta_cdf_lowersum(y, a, b, N=1000):


        xvs = jnp.linspace(0.,y, N)

        change_point = (a - 1)/(a + b - 2)

        jnp.where()

        xvs = jnp.linspace(0.,y, N)

        deltas = xvs[1:]-xvs[:-1]
        midpoints = xvs[:-1] + deltas/2.

        lys = jax.scipy.stats.beta.logpdf(midpoints,a,b)

        logdeltas = jnp.log(deltas)

        # perform rectangle rule integration
        logI = logsumexp(lys+logdeltas)


        return logI





    # from tensorflow_probability.substrates import jax as tfp
    # import scipy

    # hyperg_2F1 = tfp.math.hypergeometric.hyp2f1_small_argument

    # gammaln =  jax.scipy.special.gammaln

    # def logpbeta(x,a,b):
    #     return jnp.log(hyperg_2F1(a+b,1,a+1,x)) + a*jnp.log(x)+b*jnp.log(1-x)-jnp.log(a) -  (gammaln(a)+gammaln(b) - gammaln(a+b) )  # - lbeta(a,b)


    # def logpbeta_scipy(x,a,b):
    #     return jnp.log(scipy.special.hyp2f1(a+b,1,a+1,x)) + a*jnp.log(x)+b*jnp.log(1-x)-jnp.log(a) -  (gammaln(a)+gammaln(b) - gammaln(a+b) )  # - lbeta(a,b)


    def call_all(y,a,b, N=10000):
        #print(a , b, y, "logsumexp lowersum: ", log_beta_cdf_lowersum(y,a,b, N))
        print(a , b, y, "jax scipy: ", jnp.log(jax.scipy.stats.beta.cdf(y,a,b)))
        print(a , b, y, "tfp betainc: ", jnp.log(tfp.math.betainc(a, b, y)))
        print(a , b, y, "betainc trick: ", log_beta_cdf_trick(a, b, y))
        print(a , b, y, "logsumexp trapezium: ", log_beta_cdf_trap(y,a,b, N))
        print(a , b, y, "logsumexp rectangle: ", log_beta_cdf_rect(y,a,b, N))
        #print(a , b, y, "tfp hypergeom: ", logpbeta(y,a,b))
        #print(a , b, y, "hypergeom scipy: ", logpbeta_scipy(y,a,b))



    # for y in jnp.linspace(0.,1., 17):

    #     print("y: ", y, "logsumexp trapezium: ", log_beta_cdf(y,a,b))
    #     #print("jax.scipy: ", jnp.log(jax.scipy.stats.beta.cdf(y,a,b)))
    #     #print("tfp betainc: ", jnp.log(tfp.math.betainc(a, b, y)))

    a = 5000
    b = 10000
    y = 0.999

    call_all(y,a,b)


    a = 5000
    b = 10000
    y = 0.0001

    call_all(y,a,b)

    a = 1.2419829
    b = 16.31027743
    y= 0.94385564

    call_all(y,a,b)

    a = 0.81579644
    b = 4.17342675
    y = 0.34435819

    call_all(y,a,b)

    a = 5000
    b = 5000
    y = .00001

    call_all(y,a,b)

    a = 991.42318309
    b = 1498.45334772
    y = 0.69169976




