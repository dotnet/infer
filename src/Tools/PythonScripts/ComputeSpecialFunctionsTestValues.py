# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
"""A script to evaluate test values for special functions in high precision.

This scripts looks for .csv files in /test/Tests/Data/SpecialFunctionsValues.
These files are expected to contain sets of arguments and expected result values
for some special functions.
Whenever the script encounters a file for which it has a defined funtion,
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
from sympy import *
import os
import csv
import mpmath

mpmath.mp.pretty = true
mpmath.mp.dps = 500

def normal_cdf_moment_ratio(n, x):
    xmpf = to_mpmath(x)
    nmpf = to_mpmath(n)
    if x < 0:
        return Float(mpmath.power(2, -0.5 - nmpf / 2) * mpmath.hyperu(nmpf / 2 + 0.5, 0.5, xmpf * xmpf / 2))
    return Float(mpmath.exp(xmpf * xmpf / 4) * mpmath.pcfu(0.5 + nmpf, -xmpf))

def mpmath_normal_cdf2(x, y, r):
    """
    This function produces correct results for inputs currently present in /test/Tests/Data/SpecialFunctionsValues.
    Other inputs may fall into areas where currently present algorithms produce incorrect results and may require modifying this function.
    """
    if x == -mpmath.inf or y == -mpmath.inf:
        return mpmath.mpf('0')
    if x == mpmath.inf:
        return mpmath.ncdf(y)
    if y == mpmath.inf:
        return mpmath.ncdf(x)
    if r == mpmath.mpf('1'):
        return mpmath.ncdf(min(x, y))
    if r == mpmath.mpf('-1'):
        return mpmath.mpf('0') if x <= -y else mpmath.ncdf(x) - mpmath.ncdf(-y)

    if abs(y) > abs(x):
        z = x
        x = y
        y = z

    if r < 0:
        # phi(x,y,r) = phi(inf,y,r) - phi(-x,y,-r)
        return max(mpmath.ncdf(x) - mpmath_normal_cdf2(x, -y, -r), mpmath.mpf('0'))

    if x + y > 0:
        # phi(x,y,r) = phi(-x,-y,r) - phi(x,y,-1)
        return mpmath_normal_cdf2(-x, -y, r) + (mpmath.mpf('0') if x <= -y else mpmath.ncdf(x) - mpmath.ncdf(-y))

    def f(t):
        if abs(t) == mpmath.mpf('1'):
            return mpmath.mpf('0')
        omt2 = (1 - t) * (1 + t)
        return 1 / (2 * mpmath.pi * mpmath.sqrt(omt2)) * mpmath.exp(-(x * x + y * y - 2 * t * x * y) / (2 * omt2))
    result, err = mpmath.quad(f, [-1, r], error=True)
    
    if mpmath.mpf('1e50') * abs(err) > abs(result):
        print(f"Suspiciously big error when evaluating an integral for normal_cdf2({x}, {y}, {r}).")
        print(f"Integral: {result}")
        print(f"Integral error estimate: {err}")
    return result

def normal_cdf2(x, y, r):
    result = mpmath_normal_cdf2(to_mpmath(x), to_mpmath(y), to_mpmath(r))
    return Float(result)

def normal_cdf_ln2(x, y, r):
    # Some bad cases present in test data that were computed separately,
    # because the routine here can not provide sufficient accuracy
    if x == Float('-0.299999999999999988897769753748434595763683319091796875') and y == Float('-0.299999999999999988897769753748434595763683319091796875') and r == Float('-0.99999899999999997124433548378874547779560089111328125'):
        return Float('-90020.4997847132695760821045836463571450496491084868602301611')
    if x == Float('-0.1000000000000000055511151231257827021181583404541015625') and y == Float('-0.1000000000000000055511151231257827021181583404541015625'):
        if r == Float('-0.99999899999999997124433548378874547779560089111328125'):
            return Float('-10018.3026957438325237319456255040365689256268122985682785824')
        if r == Float('-0.99999000000000004551026222543441690504550933837890625'):
            return Float('-1014.85016355878529115329386335426754801497185281882358946348')
        if r == Float('-0.9999000000000000110134124042815528810024261474609375'):
            return Float('-111.409512020775362450645082754211442935050576861888726532')
    result = mpmath.ln(mpmath_normal_cdf2(to_mpmath(x), to_mpmath(y), to_mpmath(r)))
    return Float(result)

def logistic_gaussian(m, v):
    if m == oo:
        if v == oo:
            return oo
        return Float('1.0')
    if v == oo:
        return Float('0.5')
    mmpf = to_mpmath(m)
    vmpf = to_mpmath(v)
    logEpsilon = log(mpmath.mpf('1e-500'))
    if 2*mmpf + 4*vmpf < logEpsilon:
        return Float(exp(mmpf + vmpf/2) * (1 - exp(mmpf + 1.5 * vmpf) * (1 - exp(mmpf + 2.5 * vmpf))))
    # The integration routine below is obtained by substituting x = atanh(t)*sqrt(v)
    # into the definition of logistic_gaussian
    #
    # f = lambda x: mpmath.exp(-(x - mmpf) * (x - mmpf) / (2 * vmpf)) / (1 + mpmath.exp(-x))
    # result = 1 / mpmath.sqrt(2 * mpmath.pi * vmpf) * mpmath.quad(f, [-mpmath.inf, mpmath.inf])
    #
    # Such substitution makes mpmath.quad call much faster.
    tanhm = mpmath.tanh(mmpf)
    # Not really a precise threshold, but fine for our data
    if tanhm == mpmath.mpf('1.0'):
        return Float('1.0')
    sqrtv = mpmath.sqrt(vmpf)
    misqrtv = mmpf/sqrtv
    def f(t): 
        x = mpmath.atanh(t)
        return mpmath.exp(-(x - misqrtv) ** 2 / 2) / (1 + mpmath.exp(-x*sqrtv)) / (1 - t * t)
    coef = 1 / mpmath.sqrt(2 * mpmath.pi)
    int, err = mpmath.quad(f, [-1, 0, 1], error=True)
    result = coef * int
    if mpmath.mpf('1e50') * abs(err) > abs(int):
        print(f"Suspiciously big error when evaluating an integral for logistic_gaussian({m}, {v}).")
        print(f"Integral: {int}")
        print(f"integral error estimate: {err}")
        print(f"Coefficient: {coef}")
        print(f"Result (Coefficient * Integral): {result}")
    return Float(result)

def logistic_gaussian_deriv(m, v):
    if m.is_infinite or v.is_infinite:
        return Float('0.0')
    mmpf = to_mpmath(m)
    vmpf = to_mpmath(v)
    # The integration routine below is obtained by substituting x = atanh(t)
    # into the definition of logistic_gaussian'
    #
    # f = lambda x: mpmath.exp(-(x - mmpf) * (x - mmpf) / (2 * vmpf)) / ((1 + mpmath.exp(-x)) * (1 + mpmath.exp(x)))
    # result = 1 / mpmath.sqrt(2 * mpmath.pi * vmpf) * mpmath.quad(f, [-mpmath.inf, mpmath.inf])
    #
    # Such substitution makes mpmath.quad call much faster.
    def f(t):
        one_minus_t_squared = 1 - t * t
        return mpmath.exp(-(mpmath.atanh(t) - mmpf) ** 2 / (2 * vmpf)) / (one_minus_t_squared + mpmath.sqrt(one_minus_t_squared))
    coef = mpmath.mpf('0.5') / mpmath.sqrt(2 * mpmath.pi * vmpf)
    int, err = mpmath.quad(f, [-1, 1], error=True)
    result = coef * int
    if mpmath.mpf('1e50') * abs(err) > abs(int):
        print(f"Suspiciously big error when evaluating an integral for logistic_gaussian'({m}, {v}).")
        print(f"Integral: {int}")
        print(f"integral error estimate: {err}")
        print(f"Coefficient: {coef}")
        print(f"Result (Coefficient * Integral): {result}")
    return Float(result)

def logistic_gaussian_deriv2(m, v):
    if m.is_infinite or v.is_infinite:
        return Float('0.0')
    mmpf = to_mpmath(m)
    vmpf = to_mpmath(v)
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
        sqrt_one_minus_t_squared = mpmath.sqrt(one_minus_t_squared)
        return mpmath.exp(-(mpmath.atanh(t) - mmpf) ** 2 / (2 * vmpf)) * (one_minus_t - sqrt_one_minus_t_squared) / ((one_minus_t_squared + sqrt_one_minus_t_squared) * (one_minus_t + sqrt_one_minus_t_squared))
    coef = mpmath.mpf('0.5') / mpmath.sqrt(2 * mpmath.pi * vmpf)
    int, err = mpmath.quad(f, [-1, 1], error=True)
    result = coef * int
    if mpmath.mpf('1e50') * abs(err) > abs(int):
        print(f"Suspiciously big error when evaluating an integral for logistic_gaussian''({m}, {v}).")
        print(f"Integral: {int}")
        print(f"integral error estimate: {err}")
        print(f"Coefficient: {coef}")
        print(f"Result (Coefficient * Integral): {result}")
    return Float(result)

def normal_cdf(x):
    return erfc(-x / sqrt(S(2))) / 2

def normal_pdf_ln(x):
    return -x * x / 2 - log(sqrt(2 * pi))

def normal_cdf_integral(x, y, r):
    if x == -oo or y == -oo:
        return Float('0.0')
    if x == oo:
        return oo
    if y == oo:
        result = normal_cdf2(x, y, r)
        if x > 0:
            return result * x + exp(normal_pdf_ln(x) - log(normal_cdf(x)))
        else:
            return result * normal_cdf_moment_ratio(Float('1.0'), x) * exp(normal_pdf_ln(x) - log(normal_cdf(x)))
    if r == S(1):
        if x <= y:
            return normal_cdf_moment_ratio(Float('1.0'), x) * exp(normal_pdf_ln(x))
        else:
            npdfy = exp(normal_pdf_ln(y))
            return (normal_cdf_moment_ratio(Float('1.0'), y) + (x - y) * normal_cdf(y) / npdfy) * npdfy
    if r == S(-1):
        if x + y <= 0:
            return S(0)
        else:
            return x * (normal_cdf(y) - normal_cdf(-x)) + exp(normal_pdf_ln(x)) - exp(normal_pdf_ln(y))

    # This area separation works well for inputs currently present in /test/Tests/Data/SpecialFunctionsValues
    # Other inputs may require making this more accurate
    if x > 0 and y > 0 and 1 + r < Float('1e-12'):
        return normal_cdf_integral(x, y, S(-1)) - normal_cdf_integral(-x, -y, r)
    sqrtomr2 = sqrt((1 - r) * (1 + r))
    return x * normal_cdf2(x, y, r) + exp(normal_pdf_ln(x) + log(normal_cdf((y - r * x) / sqrtomr2))) + r * exp(normal_pdf_ln(y) + log(normal_cdf((x - r * y) / sqrtomr2)))

def normal_cdf_integral_ratio(x, y, r):
    # Some bad cases present in test data that were computed separately,
    # because the routine here can not provide sufficient accuracy
    if x == Float('-39062.4923802060075104236602783203125') and y == Float('39062.5011106818928965367376804351806640625') and r == Float('-0.9999998333405668571316482484689913690090179443359375'):
        return Float('0.000025600004960154713498351213672772546931354593117186546685502492546785021612')
    # Returned values in cases below are imprecise, because their evaluation requires values to be represented as unevaluated exponents
    # (otherwise both normal cdf and its integral end up being zeroes even in mpfr).
    # Standard libraries don't have that, so for now values produced by Infer.Net itself are used.
    if x == Float('-824.4368021638800883010844700038433074951171875') and y == Float('-23300.71373148090788163244724273681640625') and r == Float('-0.9991576459172382129736433853395283222198486328125'):
        return Float('6.9859450855259114E-08')
    if x == Float('790.8036889243788891690201126039028167724609375') and y == Float('-1081777102.232640743255615234375') and r == Float('-0.9458744064347397451086862929514609277248382568359375'):
        return Float('1.0293108592794333E-10')
    if x == Float('790.8036889243788891690201126039028167724609375') and y == Float('-1081776354979.671875') and r == Float('-0.9458744064347397451086862929514609277248382568359375'):
        return Float('1.0293107755790882E-13')
    int_z = normal_cdf_integral(x, y, r)
    if int_z == S(0):
        return int_z
    z = normal_cdf2(x, y, r)
    return int_z / z

def beta_cdf(x, a, b):
    if x <= S(0):
        return Float('0.0')
    if x >= S(1):
        return Float('1.0')
    result = mpmath.betainc(to_mpmath(a), to_mpmath(b), 0, to_mpmath(x), regularized=True)
    return Float(result)

pair_info = {
    'BesselI.csv': besseli,
    'BetaCdf.csv': beta_cdf,
    'Digamma.csv': digamma,
    'Erfc.csv': erfc,
    'ExpMinus1.csv': lambda x: exp(x) - 1,
    'ExpMinus1RatioMinus1RatioMinusHalf.csv': lambda x: ((exp(x) - 1) / x - 1) / x - S(1) / 2 if x != S(0) else S(0),
    'Gamma.csv': gamma,
    'GammaLn.csv': loggamma,
    'GammaLower.csv': lambda s, x: lowergamma(s, x) / gamma(s) if s != oo else S(0),
    'GammaUpper.csv': uppergamma,
    'GammaUpperRegularized.csv': lambda s, x: 1 - (lowergamma(s, x) / gamma(s) if s != oo else S(0)),
    'GammaUpperScale.csv' : lambda s, x: x ** s * exp(-x) / gamma(s),
    'Log1MinusExp.csv': lambda x: log(1 - exp(x)),
    'Log1Plus.csv': lambda x: log(1 + x),
    'LogExpMinus1.csv': lambda x: log(exp(x) - 1),
    'Logistic.csv': lambda x: 1 / (1 + exp(-x)),
    'logisticGaussian.csv': logistic_gaussian,
    'logisticGaussianDeriv.csv': logistic_gaussian_deriv,
    'logisticGaussianDeriv2.csv': logistic_gaussian_deriv2,
    'LogisticLn.csv': lambda x: -log(1 + exp(-x)),
    'LogSumExp.csv': lambda x, y: log(exp(x) + exp(y)),
    'NormalCdf.csv': normal_cdf,
    'NormalCdf2.csv': normal_cdf2,
    'NormalCdfIntegral.csv': normal_cdf_integral,
    'NormalCdfIntegralRatio.csv': normal_cdf_integral_ratio,
    'NormalCdfInv.csv': lambda x: -sqrt(S(2)) * erfinv(1 - 2 * x),
    'NormalCdfLn.csv': lambda x: log(normal_cdf(x)),
    'NormalCdfLn2.csv': normal_cdf_ln2,
    'NormalCdfLogit.csv': lambda x: log(normal_cdf(x)) - log(normal_cdf(-x)),
    'NormalCdfMomentRatio.csv': normal_cdf_moment_ratio,
    'NormalCdfRatioLn2.csv': None, # All test cases are pathological and need to be computed separately.
    'Tetragamma.csv': lambda x: polygamma(2, x),
    'Trigamma.csv': trigamma,
    'ulp.csv': None
    }

def to_mpmath(x):
    if x == -oo:
        return -mpmath.inf
    if x == oo:
        return mpmath.inf
    return x._to_mpmath(mpmath.mp.dps)

def float_str_csharp_to_python(s):
    return s.replace('NaN', 'nan').replace('Infinity', 'inf')

def float_str_python_to_csharp(s):
    return s.replace('nan', 'NaN').replace('inf', 'Infinity').replace('oo', 'Infinity')

dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'test', 'Tests', 'data', 'SpecialFunctionsValues')
with os.scandir(dir) as it:
    for entry in it:
        if entry.name.endswith('.csv') and entry.is_file() and entry.name == "logisticGaussian.csv":
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
                    newrow = dict(row)
                    args = []
                    for i in range(arg_count):
                        args.append(Float(float_str_csharp_to_python(row[f'arg{i}'])))
                    result_in_file = row['expectedresult']
                    if result_in_file == 'Infinity' or result_in_file == '-Infinity' or result_in_file == 'NaN' or len(newrows) > 5:
                        newrow['expectedresult'] = result_in_file
                    else:
                        print(f'{entry.name}{args}')
                        try:
                            result = f(*args).evalf(50, maxn=500)
                            print(result)
                            if abs(result) < Float('1e-20000'):
                                result = Float('0')
                        except ValueError:
                            print(f'ValueError for args {args}. Setting result to NaN.')
                            result = Float('nan')
                        newrow['expectedresult'] = float_str_python_to_csharp(str(result))
                    newrows.append(newrow)

            with open(entry.path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
                writer.writeheader()
                writer.writerows(newrows)