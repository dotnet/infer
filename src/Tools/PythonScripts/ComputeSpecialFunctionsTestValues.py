# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
"""A script to evaluate test values for special functions in high precision.

This scripts looks for .csv files in /test/Tests/Data/SpecialFunctionsValues.
These files are expected to contain sets of arguments and expected result values
for some special functions.
Whenever the script encounters a file for which it has a defined function,
it evaluates that function for every set of arguments present in that file
and replaces the expected result in the file with the one it computed,
except for Infinite or NaN results, which are preserved.

.csv files are expected to have the header of the form
arg0,arg1,...,argN,expectedresult
use comma as a value separator, dot as a decimal separator, and
"Infinity", "-Infinity", and "NaN" to designate the corresponding values.

The correspondence between file names and functions is set in the pair_info
dictionary within the script.

To add a new test case, add a new row to the csv file using zero for the expectedresult.
Then run this script to replace the dummy value.
"""
from __future__ import division
import os
import csv
from mpmath import *
import time

mp.pretty = True
mp.dps = 500
output_dps = 50


def normal_cdf_moment_ratio(n, x):
    if x < 0:
        return power(2, -0.5 - n / 2) * hyperu(n / 2 + 0.5, 0.5, x * x / 2)
    return exp(x * x / 4) * pcfu(0.5 + n, -x)

def normal_cdf2(x, y, r):
    """
    This function produces correct results for inputs currently present in /test/Tests/Data/SpecialFunctionsValues.
    Other inputs may fall into areas where currently present algorithms produce incorrect results and may require modifying this function.
    """
    if x == -inf or y == -inf:
        return mpf('0')
    if x == inf:
        return ncdf(y)
    if y == inf:
        return ncdf(x)
    if r == mpf('1'):
        return ncdf(min(x, y))
    if r == mpf('-1'):
        if x <= -y:
            return mpf('0') 
        elif x > y:
            return ncdf(y) - ncdf(-x)
        else:
            return ncdf(x) - ncdf(-y)

    if abs(y) > abs(x):
        z = x
        x = y
        y = z

    # Avoid quadrature with r < 0 since it is sometimes inaccurate.
    if r < 0 and x - y < 0:
        # phi(x,y,r) = phi(inf,y,r) - phi(-x,y,-r)
        # phi(x,y,r) = phi(x,inf,r) - phi(x,-y,-r)
        return ncdf(x) - normal_cdf2(x, -y, -r)

    if x > 0 and -x + y <= 0:
        return ncdf(y) - normal_cdf2(-x,y,-r)

    if x + y > 0:
        # phi(x,y,r) = phi(-x,-y,r) + phi(x,y,-1)
        return normal_cdf2(-x, -y, r) + normal_cdf2(x,y,-1)

    def f(t):
        if abs(t) == mpf('1'):
            # When t = -1, (x*x+y*y-2*t*x*y) = (x+y)^2 >= 0
            # When t = 1, (x*x+y*y-2*t*x*y) = (x-y)^2 >= 0
            return mpf('0')
        omt2 = (1 - t) * (1 + t)
        return 1 / (2 * pi * sqrt(omt2)) * exp(-(x * x + y * y - 2 * t * x * y) / (2 * omt2))

    omr2 = (1+r)*(1-r)
    ymrx = y - r*x
    def f2(t):
        return npdf(t - x) * normal_cdf((ymrx + r*t)/omr2)

    # This integral excludes normal_cdf2(x,y,-1)
    # which will be zero when x+y <= 0
    result, err = safe_quad(f, [-1, r])
    
    if mpf(10)**output_dps * abs(err) > abs(result):
        result, err = safe_quad(f2, [0, inf])
        if mpf(10)**output_dps * abs(err) > abs(result):
            print(f"Suspiciously big error when evaluating an integral for normal_cdf2({nstr(x)}, {nstr(y)}, {nstr(r)}).")
            print(f"Integral: {nstr(result)}")
            print(f"Integral error estimate: {nstr(err)}")
    return result

def safe_quad(f, points):
    verbose=False
    # get a quick estimate of the result
    estimate = quad(f, points, maxdegree=1, verbose=verbose)
    if verbose:
        print(f"Rescaling integrand by {nstr(1/estimate)}")
    result, err = quad(lambda x: f(x)/estimate, points, error=True, verbose=verbose)
    result *= estimate
    err *= estimate
    if mpf(10)**output_dps * abs(err) > abs(result):
        estimate = result
        if verbose:
            print(f"Rescaling integrand by {nstr(1/estimate)}")
        result, err = quad(lambda x: f(x)/estimate, points, error=True, verbose=verbose)
        result *= estimate
        err *= estimate
    return result, err

def normal_cdf2_ln(x, y, r):
    return ln(normal_cdf2(x, y, r))

def normal_cdf2_ratio_ln(x, y, r, sqrtomr2):
    if sqrtomr2 < 0.618:
        omr2 = sqrtomr2*sqrtomr2
        r = sign(r)*sqrt(1 - omr2)
    else:
        omr2 = 1-r*r
    return normal_cdf2_ln(x, y, r) + (x*x+y*y-2*r*x*y)/2/omr2 + log(2*pi)

def logistic_gaussian(m, v):
    if m == inf:
        if v == inf:
            return inf
        return mpf('1.0')
    if v == inf:
        return mpf('0.5')
    logEpsilon = log(eps)
    if 2*m + 4*v < logEpsilon:
        return mpf(exp(m + v/2) * (1 - exp(m + 1.5 * v) * (1 - exp(m + 2.5 * v))))
    tanhm = tanh(m)
    # Not really a precise threshold, but fine for our data
    if tanhm == mpf('1.0'):
        return tanhm
    # The integration routine below is obtained by substituting x = atanh(t)*sqrt(v)
    # into the definition of logistic_gaussian
    #
    # f = lambda x: mpmath.exp(-(x - mmpf) * (x - mmpf) / (2 * vmpf)) / (1 + mpmath.exp(-x))
    # result = 1 / mpmath.sqrt(2 * mpmath.pi * vmpf) * mpmath.quad(f, [-mpmath.inf, mpmath.inf])
    #
    # Such substitution makes mpmath.quad call much faster.
    # mpmath.quad uses exponential spacing between quadrature points, so we want the transformation to grow like log(x).
    sqrtv = sqrt(v)
    misqrtv = m/sqrtv
    scale = max(10, m + sqrtv)/sqrtv
    def f(t): 
        x = scale*atanh(t)
        return exp(-(x - misqrtv) ** 2 / 2) / (1 + exp(-x*sqrtv)) / (1 - t * t)
    coef = scale / sqrt(2 * pi)
    points = [-1, 0, 1]
    int, err = safe_quad(f, points)
    result = coef * int
    if mpf(10)**output_dps * abs(err) > abs(int):
        print(f"Suspiciously big error when evaluating an integral for logistic_gaussian({nstr(m)}, {nstr(v)}).")
        print(f"Integral: {nstr(int)}")
        print(f"integral error estimate: {nstr(err)}")
        print(f"Coefficient: {nstr(coef)}")
        print(f"Result (Coefficient * Integral): {nstr(result)}")
    return result

def logistic_gaussian_deriv(m, v):
    if m == inf or m == -inf or v == inf:
        return mpf('0.0')
    # The integration routine below is obtained by substituting x = atanh(t)
    # into the definition of logistic_gaussian'
    #
    # f = lambda x: mpmath.exp(-(x - mmpf) * (x - mmpf) / (2 * vmpf)) / ((1 + mpmath.exp(-x)) * (1 + mpmath.exp(x)))
    # result = 1 / mpmath.sqrt(2 * mpmath.pi * vmpf) * mpmath.quad(f, [-mpmath.inf, mpmath.inf])
    #
    # Such substitution makes mpmath.quad call much faster.
    def f(t):
        one_minus_t_squared = 1 - t * t
        return exp(-(atanh(t) - m) ** 2 / (2 * v)) / (one_minus_t_squared + sqrt(one_minus_t_squared))
    coef = 0.5 / sqrt(2 * pi * v)
    int, err = safe_quad(f, [-1, 1])
    result = coef * int
    if mpf(10)**output_dps * abs(err) > abs(int):
        print(f"Suspiciously big error when evaluating an integral for logistic_gaussian'({m}, {v}).")
        print(f"Integral: {int}")
        print(f"integral error estimate: {err}")
        print(f"Coefficient: {coef}")
        print(f"Result (Coefficient * Integral): {result}")
    return result

def logistic_gaussian_deriv2(m, v):
    if m == inf or m == -inf or v == inf or m == mpf(0):
        return mpf(0)
    # The integration routine below is obtained by substituting x = atanh(t)
    # into the definition of logistic_gaussian''
    #
    # def f(x):
    #     expx = mpmath.exp(x)
    #     one_plus_expx = 1 + expx
    #     return mpmath.exp(-(x - mmpf) * (x - mmpf) / (2 * vmpf)) * (1 - expx) / ((1 + mpmath.exp(-x)) * one_plus_expx * one_plus_expx)
    # coef = 1 / mpmath.sqrt(2 * mpmath.pi * vmpf)
    # int = mpmath.quad(f, [-mpmath.inf, mpmath.inf])
    # result = coef * int
    #
    # Such substitution makes mpmath.quad call much faster.
    def f(t):
        one_minus_t = 1 - t
        one_minus_t_squared = 1 - t * t
        sqrt_one_minus_t_squared = sqrt(one_minus_t_squared)
        return exp(-(atanh(t) - m) ** 2 / (2 * v)) * (one_minus_t - sqrt_one_minus_t_squared) / ((one_minus_t_squared + sqrt_one_minus_t_squared) * (one_minus_t + sqrt_one_minus_t_squared))
    coef = 0.5 / sqrt(2 * pi * v)
    int, err = safe_quad(f, [-1, 1])
    result = coef * int
    if mpf(10)**output_dps * abs(err) > abs(int):
        print(f"Suspiciously big error when evaluating an integral for logistic_gaussian''({m}, {v}).")
        print(f"Integral: {nstr(int)}")
        print(f"integral error estimate: {nstr(err)}")
        print(f"Coefficient: {nstr(coef)}")
        print(f"Result (Coefficient * Integral): {nstr(result)}")
    return result

def normal_cdf(x):
    """
    An alternate way of computing ncdf that avoids the bugs in ncdf
    """
    return 0.5 * gammainc(0.5, x * x / 2, inf) / gamma(0.5)

def normal_pdf_ln(x):
    return -x * x / 2 - log(sqrt(2 * pi))

def normal_cdf_integral(x, y, r):
    if x == -inf or y == -inf:
        return mpf('0.0')
    if x == inf:
        return inf
    if y == inf:
        result = normal_cdf2(x, y, r)
        if x > 0:
            return result * x + exp(normal_pdf_ln(x) - log(ncdf(x)))
        else:
            return result * normal_cdf_moment_ratio(mpf('1.0'), x) * exp(normal_pdf_ln(x) - log(ncdf(x)))
    if r == mpf(1):
        if x <= y:
            return normal_cdf_moment_ratio(mpf('1.0'), x) * exp(normal_pdf_ln(x))
        else:
            npdfy = exp(normal_pdf_ln(y))
            return (normal_cdf_moment_ratio(mpf('1.0'), y) + (x - y) * ncdf(y) / npdfy) * npdfy
    if r == mpf(-1):
        if x + y <= 0:
            return mpf(0)
        else:
            return x * normal_cdf2(x, y, r) + npdf(x) - npdf(y)

    # This area separation works well for inputs currently present in /test/Tests/Data/SpecialFunctionsValues
    # Other inputs may require making this more accurate
    if x > 0 and y > 0 and 1 + r < mpf('1e-12'):
        return normal_cdf_integral(x, y, -1) - normal_cdf_integral(-x, -y, r)
    omr2 = (1-r)*(1+r)
    sqrtomr2 = sqrt(omr2)
    # This is accurate when x >= 0 and r >= 0
    if True: #x >= 0 and r >= 0:
        return x * normal_cdf2(x, y, r) + exp(normal_pdf_ln(x) + log(ncdf((y - r * x) / sqrtomr2))) + r * exp(normal_pdf_ln(y) + log(ncdf((x - r * y) / sqrtomr2)))
    # try quadrature on the integral definition
    def f(t):
        return t * npdf(t - x) * normal_cdf((y - r*(x-t))/omr2)
    result, err = safe_quad(f, [0, inf])
    if mpf(10)**output_dps * abs(err) > abs(result):
        print(f"Suspiciously big error when evaluating an integral for normal_cdf_integral({x}, {y}, {r}).")
        print(f"Integral: {nstr(result)}")
        print(f"integral error estimate: {nstr(err)}")
    return result



def normal_cdf_integral_ratio(x, y, r):
    int_z = normal_cdf_integral(x, y, r)
    if int_z == mpf(0):
        return int_z
    z = normal_cdf2(x, y, r)
    return int_z / z

def beta_cdf(x, a, b):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    return betainc(a, b, 0, x, regularized=True)

pair_info = {
    'BesselI.csv': besseli,
    'BetaCdf.csv': beta_cdf,
    'Digamma.csv': digamma,
    'Erfc.csv': erfc,
    'ExpMinus1.csv': expm1,
    'ExpMinus1RatioMinus1RatioMinusHalf.csv': lambda x: ((exp(x) - 1) / x - 1) / x - 0.5 if x != mpf(0) else mpf(0),
    'Gamma.csv': gamma,
    'GammaLn.csv': loggamma,
    'GammaLnSeries.csv': lambda x: loggamma(x) - (x-0.5)*log(x) + x - 0.5*log(2*pi),
    'GammaLower.csv': lambda s, x: gammainc(s, 0, x, regularized=True) if s != inf else mpf(0),
    'GammaUpper.csv': lambda s, x: gammainc(s, x, inf),
    'GammaUpperRegularized.csv': lambda s, x: gammainc(s, x, inf, regularized=True) if s != inf else mpf(1),
    'GammaUpperScale.csv' : lambda s, x: x ** s * exp(-x) / gamma(s),
    'Log1MinusExp.csv': lambda x: log(1 - exp(x)),
    'Log1Plus.csv': log1p,
    'LogExpMinus1.csv': lambda x: log(exp(x) - 1),
    'Logistic.csv': lambda x: 1 / (1 + exp(-x)),
    'logisticGaussian.csv': logistic_gaussian,
    'logisticGaussianDeriv.csv': logistic_gaussian_deriv,
    'logisticGaussianDeriv2.csv': logistic_gaussian_deriv2,
    'LogisticLn.csv': lambda x: -log(1 + exp(-x)),
    'LogSumExp.csv': lambda x, y: log(exp(x) + exp(y)),
    'NormalCdf.csv': ncdf,
    'NormalCdf2.csv': normal_cdf2,
    'NormalCdfIntegral.csv': normal_cdf_integral,
    'NormalCdfIntegralRatio.csv': normal_cdf_integral_ratio,
    'NormalCdfInv.csv': lambda x: -sqrt(mpf(2)) * erfinv(1 - 2 * x),
    'NormalCdfLn.csv': lambda x: log(ncdf(x)),
    'NormalCdfLn2.csv': normal_cdf2_ln,
    'NormalCdfLogit.csv': lambda x: log(ncdf(x)) - log(ncdf(-x)),
    'NormalCdfMomentRatio.csv': normal_cdf_moment_ratio,
    'NormalCdfRatioLn2.csv': normal_cdf2_ratio_ln,
    'Tetragamma.csv': lambda x: polygamma(2, x),
    'Trigamma.csv': lambda x: polygamma(1, x),
    'XMinusLog1Plus.csv': lambda x: x - log(1+x),
    }

def float_str_csharp_to_python(s):
    return s.replace('NaN', 'nan').replace('Infinity', 'inf')

def float_str_python_to_csharp(s):
    return s.replace('nan', 'NaN').replace('inf', 'Infinity').replace('inf', 'Infinity')

dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'test', 'Tests', 'data', 'SpecialFunctionsValues')
with os.scandir(dir) as it:
    for entry in it:
        if entry.name.endswith('.csv') and entry.is_file():
            print(f'Processing {entry.name}...')
            if entry.name not in pair_info.keys() or pair_info[entry.name] == None:
                print("Don't know how to process. Skipping.")
                continue
            f = pair_info[entry.name]
            with open(entry.path) as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',')
                fieldnames = reader.fieldnames
                arg_count = len(fieldnames) - 1
                newrows = []
                for row in reader:
                    if entry.name == 'NormalCdfRatioLn2.csv':
                        sqrtomr2 = mpf(float_str_csharp_to_python(row['arg3']))
                        r = mpf(float_str_csharp_to_python(row['arg2']))
                        if sqrtomr2 < 0.618:
                            row['arg2'] = nstr(sign(r)*sqrt(1-sqrtomr2*sqrtomr2), output_dps)
                    newrow = dict(row)
                    args = []
                    for i in range(arg_count):
                        args.append(mpf(float_str_csharp_to_python(row[f'arg{i}'])))
                    result_in_file = row['expectedresult']
                    verbose = True
                    if result_in_file == 'Infinity' or result_in_file == '-Infinity' or result_in_file == 'NaN':
                        newrow['expectedresult'] = result_in_file
                    else:
                        try:
                            if verbose:
                                print(f'{entry.name}{args}')
                                startTime = time.time()
                            result = f(*args)
                            if verbose:
                                elapsed = time.time() - startTime
                                print(f'({elapsed} seconds elapsed)')
                                nprint(result, output_dps)
                        except ValueError:
                            print(f'ValueError for args {args}. Setting result to NaN.')
                            result = mpf('nan')
                        newrow['expectedresult'] = float_str_python_to_csharp(nstr(result, output_dps))
                    newrows.append(newrow)

            with open(entry.path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
                writer.writeheader()
                writer.writerows(newrows)