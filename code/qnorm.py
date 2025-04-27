## Translated to python/jax from R source qnorm.c :  https://cran.r-project.org/sources.html

from jax import numpy as jnp

# /* Do the boundaries exactly for q*() functions :
#  * Often  _LEFT_ = ML_NEGINF , and very often _RIGHT_ = ML_POSINF;
#  *
#  * R_Q_P01_boundaries(p, _LEFT_, _RIGHT_)  :<==>
#  *
#  *     R_Q_P01_check(p);
#  *     if (p == R_DT_0) return _LEFT_ ;
#  *     if (p == R_DT_1) return _RIGHT_;
#  *
#  * the following implementation should be more efficient (less tests):
#  */
# #define R_Q_P01_boundaries(p, _LEFT_, _RIGHT_)		\
#     if (log_p) {					\
# 	if(p > 0)					\
# 	    ML_WARN_return_NAN;				\
# 	if(p == 0) /* upper bound*/			\
# 	    return lower_tail ? _RIGHT_ : _LEFT_;	\
# 	if(p == ML_NEGINF)				\
# 	    return lower_tail ? _LEFT_ : _RIGHT_;	\
#     }							\
#     else { /* !log_p */					\
# 	if(p < 0 || p > 1)				\
# 	    ML_WARN_return_NAN;				\
# 	if(p == 0)					\
# 	    return lower_tail ? _LEFT_ : _RIGHT_;	\
# 	if(p == 1)					\
# 	    return lower_tail ? _RIGHT_ : _LEFT_;	\
#     }


#/* Use 0.5 - p + 0.5 to perhaps gain 1 bit of accuracy */
#define R_D_Lval(p)	(lower_tail ? (p) : (0.5 - (p) + 0.5))	/*  p  */

def R_D_Lval(p, lower_tail): 
    return p if lower_tail else 0.5 - p + 0.5  # p


#define R_D_Cval(p)	(lower_tail ? (0.5 - (p) + 0.5) : (p))	/*  1 - p */

def R_D_Cval(p, lower_tail):
    return 0.5 - p + 0.5 if lower_tail else p  # 1 - p

#/*#define R_DT_qIv(p)	R_D_Lval(R_D_qIv(p))		 *  p  in qF ! */
#define R_DT_qIv(p)	(log_p ? (lower_tail ? exp(p) : - expm1(p))     : R_D_Lval(p))

def R_DT_qIv(p, lower_tail, log_p):
    if log_p:
        return jnp.exp(p) if lower_tail else -jnp.expm1(p)
    else:
        return R_D_Lval(p, lower_tail)

#/*#define R_DT_CIv(p)	R_D_Cval(R_D_qIv(p))		 *  1 - p in qF */
#define R_DT_CIv(p)	(log_p ? (lower_tail ? -expm1(p) : exp(p))    : R_D_Cval(p))

def R_DT_CIv(p, lower_tail, log_p):
    if log_p:
        return -jnp.expm1(p) if lower_tail else jnp.exp(p)
    else:
        return R_D_Cval(p, lower_tail)	       

M_2PI	=	6.283185307179586476925286766559	#/* 2*pi */
M_SQRT2	=	1.414213562373095048801688724210	#/* sqrt(2) */


NAN = jnp.nan
ML_NEGINF = -jnp.inf
ML_POSINF = jnp.inf

def nontraced_qnorm5(p, mu, sigma, lower_tail, log_p):

    if (jnp.isnan(p) or jnp.isnan(mu) or jnp.isnan(sigma)):
        return p + mu + sigma
    
    #R_Q_P01_boundaries(p, ML_NEGINF, ML_POSINF);
    _LEFT_ = ML_NEGINF
    _RIGHT_ = ML_POSINF
    if (log_p):
        if(p > 0):
            return NAN
        if(p == 0): #/* upper bound*/			
            return _RIGHT_ if lower_tail else _LEFT_
        if(p == ML_NEGINF):				
            return _LEFT_ if lower_tail else _RIGHT_
        							
    else:  #/* !log_p */					
        if(p < 0 or p > 1):				
            return NAN				
        if(p == 0):					
            return _LEFT_ if lower_tail else _RIGHT_	
        if(p == 1):					
            return _RIGHT_ if lower_tail else _LEFT_
        


    if (sigma < 0):
        return NAN; 
    if (sigma == 0):
        return mu

    p_ = R_DT_qIv(p, lower_tail, log_p); #/* real lower_tail prob. p */
    q = p_ - 0.5

    #ifdef DEBUG_qnorm
    #  REprintf("qnorm(p=%10.7g, m=%g, s=%g, l.t.= %d, log= %d): q = %g\n",
    #           p, mu, sigma, lower_tail, log_p, q);
    #endif

    #   /*-- use AS 241 --- */
    #   /* double ppnd16_(double *p, long *ifault)*/
    #   /*      ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3

    #           Produces the normal deviate Z corresponding to a given lower
    #           tail area of P; Z is accurate to about 1 part in 10**16.

    #           (original fortran code used PARAMETER(..) for the coefficients
    #            and provided hash codes for checking them...)
    #   */
    if (jnp.abs(q) <= .425):
    #                      /* |p~ - 0.5| <= .425  <==> 0.075 <= p~ <= 0.925 */
        r = .180625 - q * q; # = .425^2 - q^2  >= 0
        val =  q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r + 133.14166789178437745) * r + 3.387132872796366608) / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r + 42.313330701600911252) * r + 1.)
    
    else:
    # /* closer than 0.075 from {0,1} boundary :
    #    *  r := log(p~);  p~ = min(p, 1-p) < 0.075 :  */
        if (log_p and ((lower_tail and q <= 0) or (not lower_tail and q > 0))):
            lp = p
        else:
            #lp = log((q > 0) ? R_DT_CIv(p) /* 1-p */ : p_ /* = R_DT_Iv(p) ^=  p */);
            lp = jnp.log(R_DT_CIv(p, lower_tail, log_p) if q > 0 else p_ )
        
        #// r = sqrt( - log(min(p,1-p)) )  <==>  min(p, 1-p) = exp( - r^2 ) :
        r = jnp.sqrt(-lp)
    # #ifdef DEBUG_qnorm
    #     REprintf("\t close to 0 or 1: r = %7g\n", r);
    # #endif
        if (r <= 5.):
        #{ /* <==> min(p,1-p) >= exp(-25) ~= 1.3888e-11 */
            r += -1.6
            val = (((((((r * 7.7454501427834140764e-4 +\
                        .0227238449892691845833) *\
                            r +\
                        .24178072517745061177) *\
                            r +\
                        1.27045825245236838258) *\
                            r +
                        3.64784832476320460504) *\
                        r +\
                    5.7694972214606914055) *\
                        r +\
                    4.6303378461565452959) *\
                        r +\
                    1.42343711074968357734) /\
                    (((((((r *\
                            1.05075007164441684324e-9 +\
                        5.475938084995344946e-4) *\
                            r +\
                        .0151986665636164571966) *\
                            r +\
                        .14810397642748007459) *\
                            r +\
                        .68976733498510000455) *\
                        r +\
                    1.6763848301838038494) *\
                        r +\
                    2.05319162663775882187) *\
                        r +\
                    1.)
        
        elif (r <= 27):
                # { /* p is very close to  0 or 1: r in (5, 27] :
                # *  r >   5 <==> min(p,1-p)  < exp(-25) = 1.3888..e-11
                # *  r <= 27 <==> min(p,1-p) >= exp(-27^2) = exp(-729) ~= 2.507972e-317
                # * i.e., we are just barely in the range where min(p, 1-p) has not yet underflowed to zero.
                # */
                # // Wichura, p.478: minimax rational approx R_3(t) is for 5 <= t <= 27  (t :== r)
                r += -5.
                val = (((((((r * 2.01033439929228813265e-7 +\
                            2.71155556874348757815e-5) *\
                                r +\
                            .0012426609473880784386) *\
                                r +\
                            .026532189526576123093) *\
                                r +\
                            .29656057182850489123) *\
                            r +\
                        1.7848265399172913358) *\
                            r +\
                        5.4637849111641143699) *\
                            r +\
                        6.6579046435011037772) /\
                        (((((((r *\
                                2.04426310338993978564e-15 +\
                            1.4215117583164458887e-7) *\
                                r +\
                            1.8463183175100546818e-5) *\
                                r +\
                            7.868691311456132591e-4) *\
                                r +\
                            .0148753612908506148525) *\
                            r +\
                        .13692988092273580531) *\
                            r +\
                        .59983220655588793769) *\
                            r +\
                        1.)
        
        else:
            #{ // r > 27: p is *really* close to 0 or 1 .. practically only when log_p =TRUE
            if (r >= 6.4e8):
                # // p is *very extremely* close to 0 or 1
                #// Using the asymptotical formula ("0-th order"): qn = sqrt(2*s)
                val = r * M_SQRT2
            
            else:
                s2 = -jnp.ldexp(lp, 1)    # // = -2*lp = 2s
                x2 = s2 - jnp.log(M_2PI * s2); # // = xs_1
                #if(r >= 36000.):  # <==> s >= 36000^2   use x2 = xs_1  above
                if (r < 36000.):
                    x2 = s2 - jnp.log(M_2PI * x2) - 2. / (2. + x2); #// == xs_2
                    if (r < 840.):
                        x2 = s2 - jnp.log(M_2PI * x2) + 2 * jnp.log1p(-(1 - 1 / (4 + x2)) / (2. + x2)); #// == xs_3
                        if (r < 109.):
                            # // 27 < r < 109
                            x2 = s2 - jnp.log(M_2PI * x2) +\
                                2 * jnp.log1p(-(1 - (1 - 5 / (6 + x2)) / (4. + x2)) / (2. + x2)); #// == xs_4
                            if (r < 55.):
                                # // 27 < r < 55
                                x2 = s2 - jnp.log(M_2PI * x2) +\
                                    2 * jnp.log1p(-(1 - (1 - (5 - 9 / (8. + x2)) / (6. + x2)) / (4. + x2)) / (2. + x2)); #// == xs_5
                            
                            
                
                val = jnp.sqrt(x2)
            
        if (q < 0.0):
            val = -val
    
    return mu + sigma * val

import jax
def qnorm5(p, mu, sigma, lower_tail, log_p):

    if False: # let's remove this for the moment, but it could go at the end, just overwrite values satisfying the conditions

        if (jnp.isnan(p) or jnp.isnan(mu) or jnp.isnan(sigma)):
            return p + mu + sigma
        
        #R_Q_P01_boundaries(p, ML_NEGINF, ML_POSINF);
        _LEFT_ = ML_NEGINF
        _RIGHT_ = ML_POSINF
        if (log_p):

            if lower_tail:
                jnp.where(p > 0., NAN, jnp.where(p == 0., _RIGHT_))

            if(p > 0):
                return NAN
            if(p == 0): #/* upper bound*/			
                return _RIGHT_ if lower_tail else _LEFT_
            if(p == ML_NEGINF):				
                return _LEFT_ if lower_tail else _RIGHT_
                                        
        else:  #/* !log_p */					
            if(p < 0 or p > 1):				
                return NAN				
            if(p == 0):					
                return _LEFT_ if lower_tail else _RIGHT_	
            if(p == 1):					
                return _RIGHT_ if lower_tail else _LEFT_
        


    if (sigma < 0):
        return NAN * jnp.ones_like(p) 
    if (sigma == 0):
        return mu * jnp.ones_like(p)

    p_ = R_DT_qIv(p, lower_tail, log_p); #/* real lower_tail prob. p */
    q = p_ - 0.5

    #ifdef DEBUG_qnorm
    #  REprintf("qnorm(p=%10.7g, m=%g, s=%g, l.t.= %d, log= %d): q = %g\n",
    #           p, mu, sigma, lower_tail, log_p, q);
    #endif

    #   /*-- use AS 241 --- */
    #   /* double ppnd16_(double *p, long *ifault)*/
    #   /*      ALGORITHM AS241  APPL. STATIST. (1988) VOL. 37, NO. 3

    #           Produces the normal deviate Z corresponding to a given lower
    #           tail area of P; Z is accurate to about 1 part in 10**16.

    #           (original fortran code used PARAMETER(..) for the coefficients
    #            and provided hash codes for checking them...)
    #   */


    log_p_arr = jnp.full(p.shape, log_p)
    lower_tail_arr = jnp.full(p.shape, lower_tail)


    qle425 = (jnp.abs(q) <= .425)
    condNOT425a = (log_p_arr & ((lower_tail_arr & (q <= 0)) | (~lower_tail_arr & (q > 0))))

    if False:
        # this is how it is done in R: the problem is that we are doing log (exp())
        aux = jnp.log( jnp.where(q > 0 , R_DT_CIv(p, lower_tail, log_p), p_ )  )

    else: ## this is important to allow larger scale_ab  ## but it gives inf values for q near 1
        ### ASSUME log_p = true
        assert(log_p)
        aux = jax.scipy.special.logsumexp(jnp.stack([p,jnp.zeros_like(p)]),axis=0,b=jnp.stack([-jnp.ones_like(p),+jnp.ones_like(p)]))
 
    
    #jax.debug.print("diff {diff}", diff= aux-aux2)
    #p_ = R_DT_qIv(p, lower_tail, log_p)
    
    ### ASSUME log_p = true
    assert(log_p)
    log_p_ = p if lower_tail else aux
    
    expr = jnp.where(q > 0 , aux if lower_tail else p, log_p_ )  



    lp = jnp.where(condNOT425a, p, expr)

    r = jnp.where(qle425, .180625 - q * q , jnp.sqrt(-lp) )

    #jax.debug.print("p {p} condNOT425a {condNOT425a} qle425 {qle425} lp {lp} q {q} r {r} ", r=r, q=q, qle425=qle425, lp=lp, condNOT425a=condNOT425a, p=p)
      
    val_qle425 = q * (((((((r * 2509.0809287301226727 + 33430.575583588128105) * r + 67265.770927008700853) * r + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r + 133.14166789178437745) * r + 3.387132872796366608) / (((((((r * 5226.495278852854561 + 28729.085735721942674) * r + 39307.89580009271061) * r + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r + 42.313330701600911252) * r + 1.)

    not_qle425 = ~qle425

    rle5 = (r <= 5.)
    rle27 = (r <= 27)
    rge648 = (r >= 6.4e8)

    r += jnp.where(not_qle425 & rle5 , -1.6,
                   jnp.where( not_qle425 & rle27 , -5, 0.) )
    
    val_rle5 =  (((((((r * 7.7454501427834140764e-4 +\
                        .0227238449892691845833) *\
                            r +\
                        .24178072517745061177) *\
                            r +\
                        1.27045825245236838258) *\
                            r +
                        3.64784832476320460504) *\
                        r +\
                    5.7694972214606914055) *\
                        r +\
                    4.6303378461565452959) *\
                        r +\
                    1.42343711074968357734) /\
                    (((((((r *\
                            1.05075007164441684324e-9 +\
                        5.475938084995344946e-4) *\
                            r +\
                        .0151986665636164571966) *\
                            r +\
                        .14810397642748007459) *\
                            r +\
                        .68976733498510000455) *\
                        r +\
                    1.6763848301838038494) *\
                        r +\
                    2.05319162663775882187) *\
                        r +\
                    1.)

    val_rle27 = (((((((r * 2.01033439929228813265e-7 +\
                        2.71155556874348757815e-5) *\
                            r +\
                        .0012426609473880784386) *\
                            r +\
                        .026532189526576123093) *\
                            r +\
                        .29656057182850489123) *\
                        r +\
                    1.7848265399172913358) *\
                        r +\
                    5.4637849111641143699) *\
                        r +\
                    6.6579046435011037772) /\
                    (((((((r *\
                            2.04426310338993978564e-15 +\
                        1.4215117583164458887e-7) *\
                            r +\
                        1.8463183175100546818e-5) *\
                            r +\
                        7.868691311456132591e-4) *\
                            r +\
                        .0148753612908506148525) *\
                        r +\
                    .13692988092273580531) *\
                        r +\
                    .59983220655588793769) *\
                        r +\
                    1.)

    val_rge648 = r * M_SQRT2


    def my_log_or_y(x, y=0.):
        """Return log(x) if x > 0 or y"""
        return jnp.where(x > 0., jnp.log(jnp.where(x > 0., x, 1.)), y)

    def my_log1p_or_y(x, y=0.):
        """Return log(x) if x > 0 or y"""
        return jnp.where(x > -1., jnp.log1p(jnp.where(x > -1., x, 1.)), y)

    def my_sqrt_or_y(x, y=0.):
        """Return log(x) if x > 0 or y"""
        return jnp.where(x > 0., jnp.sqrt(jnp.where(x > 0., x, 1.)), y)


    if True:
        ### s2, x2 stuff
        s2 = -jnp.ldexp(lp, 1)    # // = -2*lp = 2s
        x2 = s2 - my_log_or_y(M_2PI * s2); # // = xs_1
    
        

        try:
            x2 = jnp.where( r < 36000. , s2 - my_log_or_y(M_2PI * x2) - 2. / (2. + x2) , x2)

        except FloatingPointError:
            jax.debug.print("cond {cond} log_p {log_p} qle425 {qle425} rle5 {rle5} rle27 {rle27} rge648 {rge648} r {r} x2 {x2} s2 {s2}", r=r, x2=x2, s2=s2, qle425=qle425, rle5=rle5, rle27=rle27, rge648=rge648, log_p=p, cond = (x2<0.) & ~qle425 & ~rle5 & ~rle27 & ~rge648)
            exit()

        x2 = jnp.where( r < 840., s2 - my_log_or_y(M_2PI * x2) + 2 * my_log1p_or_y(-(1 - 1 / (4 + x2)) / (2. + x2)), x2)

        x2 = jnp.where( r < 109., s2 - my_log_or_y(M_2PI * x2) + 2 * my_log1p_or_y(-(1 - (1 - 5 / (6 + x2)) / (4. + x2)) / (2. + x2)), x2)

        x2 = jnp.where( r < 55., s2 - my_log_or_y(M_2PI * x2) + 2 * my_log1p_or_y(-(1 - (1 - (5 - 9 / (8. + x2)) / (6. + x2)) / (4. + x2)) / (2. + x2)), x2)

        val_notrge648 = my_sqrt_or_y(x2)

        val = jnp.where(qle425, val_qle425,
                        jnp.where (rle5, val_rle5, 
                                jnp.where (rle27, val_rle27, 
                                            jnp.where( rge648, val_rge648 , 
                                                        val_notrge648 )) ))


    else: #old simple version : https://github.com/SurajGupta/r-source/blob/master/src/nmath/qnorm.c

        val = jnp.where(qle425, val_qle425,
                        jnp.where (rle5, val_rle5, val_rle27))


    val = jnp.where( not_qle425 & (q < 0.0), -val, val )
    
    return mu + sigma * val



SQRT_PI_8 = jnp.sqrt(jnp.pi/8.)

def qnorm_logit(log_p):
    ones = jnp.ones_like(log_p)
    log_1mp = jax.scipy.special.logsumexp (jnp.stack([jnp.zeros_like(log_p),log_p]), axis = 0, b = jnp.stack([ones , -ones]))
    #log_1mp = jnp.log(1-jnp.exp(log_p)) 
    logit = log_p - log_1mp
    #logit = - jax.scipy.special.logsumexp(jnp.stack([-log_p, jnp.zeros_like(log_p)]), axis = 0, b = jnp.stack([ones , -ones]))
    return SQRT_PI_8 * logit



def qnorm_acklam(log_p):
    # Coefficients in rational approximations.
    a = (-3.969683028665376e+01,  2.209460984245205e+02, \
         -2.759285104469687e+02,  1.383577518672690e+02, \
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02, \
         -1.556989798598866e+02,  6.680131188771972e+01, \
         -1.328068155288572e+01 )
    c = (-7.784894002430293e-03, -3.223964580411365e-01, \
         -2.400758277161838e+00, -2.549732539343734e+00, \
          4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01, \
          2.445134137142996e+00,  3.754408661907416e+00)

    # Define break-points.
    plow  = 0.02425
    phigh = 1 - plow

    # Rational approximation for lower region:
    if log_p < jnp.log(plow):
       q  = jnp.sqrt(-2*log_p)
       return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    # Rational approximation for upper region:
    if jnp.log(phigh) < log_p:
       log_1mp =  jnp.log(1-jnp.exp(log_p))

       q  = jnp.sqrt(-2*log_1mp)
       return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    # Rational approximation for central region:
    q = p - 0.5
    r = q*q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def qnorm_trick(log_p):
    right = -jax.scipy.stats.norm.ppf(-log_p)
    left =  qnorm5(log_p, 0., 1., True, True)
    return jnp.where(jnp.isinf(right), left, right)




if False: ## some checks
    from jax import config
    config.update("jax_enable_x64", True)



    from functools import partial
    qnorm = qnorm_trick
    
    import numpy as np


    if True:
        for p in [.1,.2,.3,.4,.5,.6,.7,.8,.9]: 
            lp = jnp.log(p)
            print(lp, nontraced_qnorm5(lp, mu=0., sigma=1., lower_tail=True, log_p=True))

        log_p = -np.logspace(308,-323, 20)
        print("old: ")

        for lp  in log_p:
            print(lp, nontraced_qnorm5(lp, mu=0., sigma=1., lower_tail=True, log_p=True))


        lplast = -1.e-307
        print(lplast, nontraced_qnorm5(lplast, mu=0., sigma=1., lower_tail=True, log_p=True))


        print("new: ")
        res = qnorm(log_p)
        for r in res:
           print(r)


        print(qnorm(jnp.array(lplast)))


