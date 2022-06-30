# This code has been adapted from IncGammaBeta.jl, which I do not include as a
# dependency because it introduced incompatabilities (it has not been updated since 2016)


"""
    incgamma(a::T, x::T; upper::Bool, reg::Bool) where {T <: AbstractFloat}

For positive parameter `a` and positive integration limit `x`, computes the incomplete gamma function, as described
by the [Wikipedia article](https://en.wikipedia.org/wiki/Incomplete_gamma_function).

# Keyword arguments:
- `upper::Bool`: if `true`, the upper incomplete gamma function is returned; otherwise, the lower version is returned.
- `reg::Bool`: if `true`, the regularized incomplete gamma function is returned; otherwise, the unregularized version is returned.
"""
function incgamma(a::T, x::T; upper::Bool, reg::Bool) where {T <: AbstractFloat}

    EPS = eps(T)
    @assert a  > EPS "a should be positive"
    @assert x  > EPS "x should be positive"

    # The algorithm for numerical approximation of the incomplete gamma function
    # as proposed by [Numerical Recipes], section 6.2:
    #
    # When x > a+1, the upper gamma function can be evaluated as
    #   G(a,x) ~= e⁻ˣxᵃ / cf(a,x)
    # where 'cf(a,x) is the continued fraction defined above, its coefficients
    # 'a(i)' and 'b(i)' are implemented in 'inc_gamma_ctdfrac_closure'.
    #
    # When x < (a+1), it is more convenient to apply the following Taylor series
    # that evaluates the lower incomplete gamma function:
    #
    #                          inf
    #                         -----
    #              -x    a    \        G(a)       i
    #   g(a,x) ~= e   * x  *   >    ---------- * x
    #                         /      G(a+1+i)
    #                         -----
    #                          i=0
    #
    # Applying the following property of the gamma function:
    #
    #   G(a+1) = a * G(a)
    #
    # The Taylor series above can be further simplified to:
    #
    #                          inf
    #                         -----              i
    #              -x    a    \                 x
    #   g(a,x) ~= e   * x  *   >    -------------------------
    #                         /      a * (a+1) * ... * (a+i)
    #                         -----
    #                          i=0
    #
    # Once either a lower or an upper incomplete gamma function is evaluated,
    # the other value may be quickly obtained by applying the following
    # property of the incomplete gamma function:
    #
    #   G(a,x) + g(a,x) = G(a)
    #
    # A value of a regularized incomplete gamma function is obtained
    # by dividing g(a,x) or G(a,x) by G(a).
    #


    # This factor is common to both algorithms described above:
    ginc = exp(-x) * (x^a)

    if ( x > (a + 1) )

        # In this case evaluate the upper gamma function as described above.

        fa, fb = inc_gamma_ctdfrac_closure(a, x)
        G = ( true==upper && false==reg ? T(0) : gamma(a) )

        ginc /= ctdfrac_eval(fa, fb)


        # Apply properties of the incomplete gamma function
        # if anything else except a generalized upper incomplete
        # gamma function is desired.

        if ( false == upper )
            ginc = G - ginc
        end

        if ( true == reg )
            # Note: if a>0, gamma(a) is always greater than 0
            ginc /= G
        end
    else

        # In this case evaluate the lower gamma function as described above.

        G = ( false==upper && false==reg ? T(0) : gamma(a) )

        # Initial term of the Taylor series at i=0:
        ginc /= a
        term = ginc

        # Proceed the Taylor series for i = 1, 2, 3... until it converges:
        TOL = SPECFUN_ITER_TOL(T)
        at = a
        i = 1
        while ( abs(term) > TOL && i<SPECFUN_MAXITER )
            at += T(1)
            term *= (x / at)
            ginc += term
            i += 1
        end

        # has the series converged?
        if ( i >= SPECFUN_MAXITER )
            @warn "The aglorithm did not converge"
        end

        #
        # Apply properties of the incomplete gamma function
        # if anything else except a generalized lower incomplete
        # gamma function is requested.
        #
        if ( true == upper )
            ginc = G - ginc
        end

        if ( true == reg )
            # Note: if a>0, gamma(a) is always greater than 0
            ginc /= G
        end
    end

    return ginc
end


function inc_gamma_ctdfrac_closure(a, x)
    return i::Integer -> (-i) * (i-a),
           i::Integer -> x - a + 1 + 2*i
end

"""
Required precision for iterative algorithms,
it depends on type 't' that must be derived
from AbstractFloat.
"""
function SPECFUN_ITER_TOL(t)

    if ( Float64 == t )
        retVal = 1e-12
    elseif ( Float32 == t )
        retVal = 1f-6
    else   # t == Float16
        retVal = eps(Float16)
    end

    return retVal
end

"""
Maximum allowed number of iterations for
iterative algorithms.
"""
const SPECFUN_MAXITER = 10000


#
# Evaluates the continued fraction:
#
#                        a1
#    f = b0 + -------------------------
#                           a2
#               b1 + -----------------
#                              a3
#                     b2 + ----------
#                           b3 + ...
#
# where ai and bi are functions of 'i'.
#
# Arguments:
# * `fa::Function`: function `a(i)`
# * `fb::Function`: function `b(i)`


function ctdfrac_eval(fa::Function, fb::Function)
    #
    # The Lentz's algorithm (modified by I. J. Thompson and A. R. Barnett)
    # is applied to evaluate the continued fraction. The algorithm is
    # presented in detail in:
    #
    #   William H. Press, Saul A. Teukolsky, William T. Vetterling, Brian P. Flannery
    #   Numerical Recipes, The Art of Scientific Computing, 3rd Edition,
    #   Cambridge University Press, 2007
    #
    #   https://books.google.com/books?id=1aAOdzK3FegC&lpg=PA207&ots=3jNoK9Crpj&pg=PA208#v=onepage&f=false
    #
    # The procedure of the algorithm is as follows:
    #
    # - f0 = b0, if b0==0 then f0 = eps
    # - C0 = f0
    # - D0 = 0
    # - for j = 1, 2, 3, ...
    #   -- Dj = bj + aj * D_j-1, if Dj==0 then Dj = eps
    #   -- Cj = bj + aj / C_j-1, if Cj==0 then Cj = eps
    #   -- Dj = 1 / Dj
    #   -- Delta_j = Cj * Dj
    #   -- fj = f_j-1 * Delta_j
    #   -- if abs(Delta_j-1) < TOL then exit for loop
    # - return fj
    #


    # f0 = b0
    f = fb(0)

    T = typeof(f)
    EPS = eps(T)

    # adjust f0 to eps if necessary
    if abs(f) < EPS
        f = EPS
    end

    # c0 = f0,  d0 = 0
    c = f
    d = T(0)

    # Initially Delta should not be equal to 1
    Delta = T(0)

    TOL = SPECFUN_ITER_TOL(T)
    j = 1
    while ( abs(Delta-1) > TOL && j < SPECFUN_MAXITER )
        # obtain 'aj' and 'bj'
        a = fa(j)
        b = fb(j)

        # dj = bj + aj * d_j-1
        d = b + a * d
        # adjust dj to eps if necessary
        if ( abs(d) < EPS )
            d = EPS
        end

        # cj = bj + aj/c_j-1
        c = b + a /c
        # adjust cj to eps if necessary
        if ( abs(c) < EPS )
            c = EPS
        end

        # dj = 1 / dj
        d = T(1) / d

        # Delta_j = cj * dj
        Delta = c * d

        # fj = f_j-1 * Delta_j
        f *= Delta

        # for loop's condition will check, if abs(Delta_j-1)
        # is less than the tolerance

        j += 1
    end  # while

    # check if the algorithm has converged:
    if ( j >= SPECFUN_MAXITER )
        @warn "The aglorithm did not converge"
    end

    # ... if yes, return the fj
    return f;

end
