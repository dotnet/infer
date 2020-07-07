# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
import os
import csv
import math
from scipy import special
from numpy import nan
from mpmath import fp

math_pair_info = {
    'Erfc.csv': math.erfc,
    'ExpMinus1.csv': math.expm1,
    'Gamma.csv': math.gamma,
    'GammaLn.csv': math.lgamma,
    'Log1Plus.csv': math.log1p,
    }

scipy_pair_info = {
    'BesselI.csv': special.iv,
    'BetaCdf.csv': lambda x,a,b: special.btdtr(a,b,x),
    'Digamma.csv': special.digamma,
    'Erfc.csv': special.erfc,
    'ExpMinus1.csv': special.expm1,
    #'ExpMinus1RatioMinus1RatioMinusHalf.csv': lambda x: ((exp(x) - 1) / x - 1) / x - 0.5 if x != mpf(0) else mpf(0),
    'Gamma.csv': special.gamma,
    'GammaLn.csv': special.gammaln,
    'GammaLower.csv': special.gammainc,
    #'GammaUpper.csv': special.gammaincc,
    'GammaUpperRegularized.csv': special.gammaincc,
    #'GammaUpperScale.csv' : lambda s, x: x ** s * exp(-x) / gamma(s),
    #'Log1MinusExp.csv': lambda x: log(1 - exp(x)),
    'Log1Plus.csv': special.log1p,
    #'LogExpMinus1.csv': lambda x: log(exp(x) - 1),
    #'Logistic.csv': lambda x: 1 / (1 + exp(-x)),
    #'logisticGaussian.csv': logistic_gaussian,
    #'logisticGaussianDeriv.csv': logistic_gaussian_deriv,
    #'logisticGaussianDeriv2.csv': logistic_gaussian_deriv2,
    #'LogisticLn.csv': lambda x: -log(1 + exp(-x)),
    'LogSumExp.csv': lambda x, y: special.logsumexp([x,y]),
    #'NormalCdf.csv': ncdf,
    #'NormalCdf2.csv': normal_cdf2,
    #'NormalCdfIntegral.csv': normal_cdf_integral,
    #'NormalCdfIntegralRatio.csv': normal_cdf_integral_ratio,
    #'NormalCdfInv.csv': lambda x: -sqrt(mpf(2)) * erfinv(1 - 2 * x),
    'NormalCdfLn.csv': special.log_ndtr,
    #'NormalCdfLn2.csv': normal_cdf2_ln,
    #'NormalCdfLogit.csv': lambda x: log(ncdf(x)) - log(ncdf(-x)),
    #'NormalCdfMomentRatio.csv': normal_cdf_moment_ratio,
    #'NormalCdfRatioLn2.csv': normal_cdf2_ratio_ln,
    'Tetragamma.csv': lambda x: special.polygamma(2, x),
    'Trigamma.csv': lambda x: special.polygamma(1, x),
    }

def beta_cdf(x, a, b):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    return fp.betainc(a, b, 0, x, regularized=True)

mpmath_pair_info = {
    'BesselI.csv': fp.besseli,
    'BetaCdf.csv': beta_cdf,
    'Digamma.csv': fp.digamma,
    'Erfc.csv': fp.erfc,
    'ExpMinus1.csv': fp.expm1,
    'Gamma.csv': fp.gamma,
    'GammaLn.csv': fp.loggamma,
    'GammaLower.csv': lambda s, x: fp.gammainc(s, 0, x, regularized=True),
    'GammaUpper.csv': lambda s, x: fp.gammainc(s, x, math.inf),
    'GammaUpperRegularized.csv': lambda s, x: fp.gammainc(s, x, math.inf, regularized=True),
    'Log1Plus.csv': fp.log1p,
    'NormalCdf.csv': fp.ncdf,
    'Tetragamma.csv': lambda x: fp.polygamma(2, x),
    'Trigamma.csv': lambda x: fp.polygamma(1, x),
    }

pair_infos = {
    'math': math_pair_info,
    'scipy': scipy_pair_info,
    'mpmath': mpmath_pair_info,
    }

def readrows(path):
    rows = []
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        fieldnames = reader.fieldnames
        arg_count = len(fieldnames) - 1
        for row in reader:
            args = []
            for i in range(arg_count):
                args.append(float(row[f'arg{i}']))
            result_in_file = float(row['expectedresult'])
            rows.append([args, result_in_file])
    return rows

dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', '..', 'test', 'Tests', 'data', 'SpecialFunctionsValues')
with os.scandir(dir) as it:
    for entry in it:
        if entry.name.endswith('.csv') and entry.is_file():
            print(f'Processing {entry.name}...')
            rows = readrows(entry.path)
            for libname in pair_infos:
                pair_info = pair_infos[libname]
                if entry.name not in pair_info.keys() or pair_info[entry.name] == None:
                    #print("Don't know how to process. Skipping.")
                    continue
                f = pair_info[entry.name]
                for row in rows:
                    args = row[0]
                    result_in_file = row[1]
                    try:
                        result = f(*args)
                    except:
                        result = nan
                    if math.isnan(result) and math.isnan(result_in_file):
                        err = 0
                    elif result == result_in_file: # avoid subtracting infinities
                        err = 0
                    else:
                        err = abs(result - result_in_file)/(abs(result_in_file) + 1e-100)
                    if err > 1e-13 or math.isnan(err):
                        print(f'{libname} {entry.name}{args}\t wrong by {err}')

print('Done')