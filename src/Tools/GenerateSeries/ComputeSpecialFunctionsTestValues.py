# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
from __future__ import division
from sympy import *
import os
import csv

def tetragamma(x):
    return polygamma(2, x)

pair_info = {
    'BesselI.csv': besseli,
    'BetaCdf.csv': None,
    'Digamma.csv': digamma,
    'Erfc.csv': erfc,
    'ExpMinus1.csv': None,
    'ExpMinus1RatioMinus1RatioMinusHalf.csv': None,
    'Gamma.csv': gamma,
    'GammaLn.csv': loggamma,
    'GammaLower.csv': lowergamma,
    'GammaUpper.csv': None,
    'GammaUpperRegularized.csv': None,
    'Log1MinusExp.csv': None,
    'Log1Plus.csv': None,
    'LogExpMinus1.csv': None,
    'Logistic.csv': None,
    'logisticGaussian.csv': None,
    'logisticGaussianDeriv.csv': None,
    'logisticGaussianDeriv2.csv': None,
    'LogisticLn.csv': None,
    'LogSumExp.csv': None,
    'NormalCdf.csv': None,
    'NormalCdf2.csv': None,
    'NormalCdfIntegral.csv': None,
    'NormalCdfIntegralRatio.csv': None,
    'NormalCdfInv.csv': None,
    'NormalCdfLn.csv': None,
    'NormalCdfLn2.csv': None,
    'NormalCdfLogit.csv': None,
    'NormalCdfMomentRatio.csv': None,
    'NormalCdfRatioLn2.csv': None,
    'Tetragamma.csv': tetragamma,
    'Trigamma.csv': trigamma,
    'ulp.csv': None
    }


def float_str_csharp_to_python(s):
    return s.replace('NaN', 'nan').replace('Infinity', 'inf')

def float_str_python_to_csharp(s):
    return s.replace('nan', 'NaN').replace('inf', 'Infinity')

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
                arg_count = (len(fieldnames) - 1) // 2
                newrows = []
                for row in reader:
                    newrow = dict(row)
                    #args = map(lambda s: Float(s, 500), row.values()[arg_count : 2 * arg_count])
                    args = []
                    for i in range(arg_count):
                        args.append(Float(float_str_csharp_to_python(row[f'arg{i}exact']), 500))
                    result = f(*args).evalf(50, maxn=500)
                    newrow['expectedresult'] = float_str_python_to_csharp(str(result))
                    newrows.append(newrow)

            with open(entry.path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',')
                writer.writeheader()
                writer.writerows(newrows)