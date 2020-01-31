# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
from __future__ import division
from sympy import zeta, evalf, bernoulli, symbols, Poly, series, factorial, factorial2, S, log, exp

# configuration
decimal_precision = 24

gamma_at_2_series_length = 26
gamma_at_2_variable_name = "dx"
gamma_at_2_indent = "                    "

digamma_at_2_series_length = 26
digamma_at_2_variable_name = "dx"
digamma_at_2_indent = "                    "

digamma_asymptotic_series_length = 8
digamma_asymptotic_variable_name = "invX2"
digamma_asymptotic_indent = "                "

trigamma_at_1_series_length = 2
trigamma_at_1_variable_name = "x"
trigamma_at_1_indent = "                    "

trigamma_asymptotic_series_length = 9
trigamma_asymptotic_variable_name = "invX2"
trigamma_asymptotic_indent = "                    "

tetragamma_at_1_series_length = 2
tetragamma_at_1_variable_name = "x"
tetragamma_at_1_indent = "                    "

tetragamma_asymptotic_series_length = 9
tetragamma_asymptotic_variable_name = "invX2"
tetragamma_asymptotic_indent = "                "

gammaln_asymptotic_series_length = 7
gammaln_asymptotic_variable_name = "invX2"
gammaln_asymptotic_indent = "                    "

log_1_plus_series_length = 6
log_1_plus_variable_name = "x"
log_1_plus_indent = "                    "

log_1_minus_series_length = 5
log_1_minus_variable_name = "y"
log_1_minus_indent = "                    "

x_minus_log_1_plus_series_length = 7
x_minus_log_1_plus_variable_name = "xOverAMinus1"
x_minus_log_1_plus_indent = "                    "

exp_minus_1_series_length = 5
exp_minus_1_variable_name = "x"
exp_minus_1_indent = "                    "

exp_minus_1_ratio_minus_1_ratio_minus_half_series_length = 13
exp_minus_1_ratio_minus_1_ratio_minus_half_variable_name = "x"
exp_minus_1_ratio_minus_1_ratio_minus_half_indent = "                    "

log_exp_minus_1_ratio_series_length = 5
log_exp_minus_1_ratio_variable_name = "x"
log_exp_minus_1_ratio_indent = "                    "

normcdfln_asymptotic_series_length = 8
normcdfln_asymptotic_variable_name = "z"
normcdfln_asymptotic_indent = "                    "

def formatRealCoefficient(coefficient):
    return f"%0.{decimal_precision}g" % coefficient

def printPolyFromNumbers(varname, coefficients, indent):
    if len(coefficients) <= 1:
        print(f"{indent}{formatRealCoefficient(coefficients[0])}")
        return
    if coefficients[0] != 0.0:
        print(f"{indent}{formatRealCoefficient(coefficients[0])} +")
    last_non_zero_idx = len(coefficients) - 1
    while coefficients[last_non_zero_idx] == 0.0:
        last_non_zero_idx = last_non_zero_idx - 1
    idx = 1
    parentheses = 0
    print(indent, end='')
    while idx < last_non_zero_idx:
        print(f"{varname} * ", end='')
        if coefficients[idx] != 0.0:
            print(f"({formatRealCoefficient(coefficients[idx])} +")
            print(indent, end='')
            parentheses = parentheses + 1
        idx = idx + 1
    print(f"{varname} * {formatRealCoefficient(coefficients[last_non_zero_idx])}", end='')
    for i in range(0, parentheses):
        print(")", end='')
    print()

def formatRationalCoefficient(coefficient):
    return str(coefficient).replace("/", ".0 / ") + ".0"

def printPolyFromRationals(varname, coefficients, indent):
    if len(coefficients) <= 1:
        print(f"{indent}{formatRationalCoefficient(coefficients[0])}")
        return
    if coefficients[0] != 0:
        print(f"{indent}{formatRationalCoefficient(coefficients[0])} +")
    last_non_zero_idx = len(coefficients) - 1
    while coefficients[last_non_zero_idx] == 0:
        last_non_zero_idx = last_non_zero_idx - 1
    idx = 1
    parentheses = 0
    print(indent, end='')
    while idx < last_non_zero_idx:
        print(f"{varname} * ", end='')
        if coefficients[idx] != 0:
            print(f"({formatRationalCoefficient(coefficients[idx])} +")
            print(indent, end='')
            parentheses = parentheses + 1
        idx = idx + 1
    print(f"{varname} * {formatRationalCoefficient(coefficients[last_non_zero_idx])}", end='')
    for i in range(0, parentheses):
        print(")", end='')
    print()

def gamma_at_2_coefficient(k):
    if k == 0:
        return 0.0
    return ((-1)**(k + 1)*(zeta(k + 1) - 1)/(k + 1)).evalf(decimal_precision)

def digamma_at_2_coefficient(k):
    if k == 0:
        return 0.0
    return ((-1)**(k + 1)*(zeta(k + 1) - 1)).evalf(decimal_precision)

def digamma_asymptotic_coefficient(k):
    if k == 0:
        return 0.0
    return bernoulli(2 * k) / (2 * k)

def trigamma_at_1_coefficient(k):
    return ((-1)**k * (k + 1) * zeta(k + 2)).evalf(decimal_precision)

def trigamma_asymptotic_coefficient(k):
    if k == 0:
        return 0.0
    return bernoulli(2 * k)

def tetragamma_at_1_coefficient(k):
    return ((-1)**(k + 1) * (k + 1) * (k + 2) * zeta(k + 3)).evalf(decimal_precision)

def tetragamma_asymptotic_coefficient(k):
    if k == 0:
        return 0.0
    return -(2 * k - 1) * bernoulli(2 * k - 2)

def gammaln_asymptotic_coefficient(k):
    return bernoulli(2 * k + 2) / (2 * (k + 1) * (2 * k + 1))

def log_1_plus_coefficient(k):
    if k == S(0):
        return 0
    if k % 2 == 0:
        return S(-1) / k
    return S(1) / k

def log_1_minus_coefficient(k):
    if k == 0:
        return S(0)
    return S(-1) / k

def x_minus_log_1_plus_coefficient(k):
    if k <= 1:
        return S(0)
    if k % 2 == 0:
        return S(1) / k
    return S(-1) / k

def exp_minus_1_coefficient(k):
    if k == 0:
        return S(0)
    return S(1) / factorial(k)

def exp_minus_1_ratio_minus_1_ratio_minus_half_coefficient(k):
    if k == 0:
        return S(0)
    return S(1) / factorial(k + 2)

def get_log_exp_minus_1_ratio_coefficients(count):
    x = symbols('x')
    return list(reversed(Poly(log((exp(x) - 1) / x).series(x, 0, count).removeO()).all_coeffs()))

# Formula for mth coefficient of the normcdfln asymptotic:
# \sum_{n=1}^m (-1)^{n+m+1} / n * \sum_{l1, l2, ..., ln \in N, l1 + l2 + ... + ln = m} (2 * l1 - 1)!! * (2 * l2 - 1)!! * ... * (2 * ln - 1)!!
# Can be obtained by composing the Taylor expansion for log(1 + x) and asymtotic expansion for erfc
def normcdfln_asymptotic_coefficient(m):
    if m == 0:
        return 0
    def next(v):
        idx = 1
        while idx < len(v) and v[idx] == 0:
            idx = idx + 1
        if idx == len(v): return False
        v0 = v[0]
        v[0] = 0
        v[idx] = v[idx] - 1
        v[idx - 1] = v0 + 1
        return True
    result = S((-1)**(2 + m)) * factorial2(2 * m - 1)
    for n in range(2,m+1):
        coef = S((-1)**(n + 1 + m)) / n
        deltas = []
        for k in range(0, n):
            deltas.append(0)
        deltas[-1] = m - n
        accSum = S(0)
        while True:
            accProd = S(1)
            for delta in deltas:
                accProd = accProd * factorial2(2 * (delta + 1) - 1)
            accSum = accSum + accProd
            if not next(deltas):
                break
        result = result + coef * accSum
    return result

def main():
    print("(1) - Gamma at 2:")
    gamma_at_2_coefficients = [gamma_at_2_coefficient(k) for k in range(0, gamma_at_2_series_length)]
    printPolyFromNumbers(gamma_at_2_variable_name, gamma_at_2_coefficients, gamma_at_2_indent)

    print("(2) - DiGamma at 2:")
    digamma_at_2_coefficients = [digamma_at_2_coefficient(k) for k in range(0, digamma_at_2_series_length)]
    printPolyFromNumbers(digamma_at_2_variable_name, digamma_at_2_coefficients, digamma_at_2_indent)

    print("(3) - DiGamma asymptotic:")
    digamma_asymptotic_coefficients = [digamma_asymptotic_coefficient(k) for k in range(0, digamma_asymptotic_series_length)]
    printPolyFromRationals(digamma_asymptotic_variable_name, digamma_asymptotic_coefficients, digamma_asymptotic_indent)

    print("(4) - TriGamma at 1:")
    trigamma_at_1_coefficients = [trigamma_at_1_coefficient(k) for k in range(0, trigamma_at_1_series_length)]
    printPolyFromNumbers(trigamma_at_1_variable_name, trigamma_at_1_coefficients, trigamma_at_1_indent)

    print("(5) - TriGamma asymptotic:")
    trigamma_asymptotic_coefficients = [trigamma_asymptotic_coefficient(k) for k in range(0, trigamma_asymptotic_series_length)]
    printPolyFromRationals(trigamma_asymptotic_variable_name, trigamma_asymptotic_coefficients, trigamma_asymptotic_indent)

    print("(6) - TetraGamma at 1:")
    tetragamma_at_1_coefficients = [tetragamma_at_1_coefficient(k) for k in range(0, tetragamma_at_1_series_length)]
    printPolyFromNumbers(tetragamma_at_1_variable_name, tetragamma_at_1_coefficients, tetragamma_at_1_indent)

    print("(7) - TetraGamma asymptotic:")
    tetragamma_asymptotic_coefficients = [tetragamma_asymptotic_coefficient(k) for k in range(0, tetragamma_asymptotic_series_length)]
    printPolyFromRationals(tetragamma_asymptotic_variable_name, tetragamma_asymptotic_coefficients, tetragamma_asymptotic_indent)

    print("(8) - GammaLn asymptotic:")
    gammaln_asymptotic_coefficients = [gammaln_asymptotic_coefficient(k) for k in range(0, gammaln_asymptotic_series_length)]
    printPolyFromRationals(gammaln_asymptotic_variable_name, gammaln_asymptotic_coefficients, gammaln_asymptotic_indent)

    print("(9) - log(1 + x):")
    log_1_plus_coefficients = [log_1_plus_coefficient(k) for k in range(0, log_1_plus_series_length)]
    printPolyFromRationals(log_1_plus_variable_name, log_1_plus_coefficients, log_1_plus_indent)

    print("(10) - log(1 - x):")
    log_1_minus_coefficients = [log_1_minus_coefficient(k) for k in range(0, log_1_minus_series_length)]
    printPolyFromRationals(log_1_minus_variable_name, log_1_minus_coefficients, log_1_minus_indent)

    print("(11) - x - log(1 + x):")
    x_minus_log_1_plus_coefficients = [x_minus_log_1_plus_coefficient(k) for k in range(0, x_minus_log_1_plus_series_length)]
    printPolyFromRationals(x_minus_log_1_plus_variable_name, x_minus_log_1_plus_coefficients, x_minus_log_1_plus_indent)

    print("(12) - exp(x) - 1:")
    exp_minus_1_coefficients = [exp_minus_1_coefficient(k) for k in range(0, exp_minus_1_series_length)]
    printPolyFromRationals(exp_minus_1_variable_name, exp_minus_1_coefficients, exp_minus_1_indent)

    print("(13) - ((exp(x) - 1) / x - 1) / x - 0.5:")
    exp_minus_1_ratio_minus_1_ratio_minus_half_coefficients = [exp_minus_1_ratio_minus_1_ratio_minus_half_coefficient(k) for k in range(0, exp_minus_1_ratio_minus_1_ratio_minus_half_series_length)]
    printPolyFromRationals(exp_minus_1_ratio_minus_1_ratio_minus_half_variable_name, exp_minus_1_ratio_minus_1_ratio_minus_half_coefficients, exp_minus_1_ratio_minus_1_ratio_minus_half_indent)

    print("(14) - log(exp(x) - 1) / x:")
    log_exp_minus_1_ratio_coefficients = get_log_exp_minus_1_ratio_coefficients(log_exp_minus_1_ratio_series_length)
    printPolyFromRationals(log_exp_minus_1_ratio_variable_name, log_exp_minus_1_ratio_coefficients, log_exp_minus_1_ratio_indent)

    print("(15) - normcdfln asymptotic:")
    normcdfln_asymptotic_coefficients = [normcdfln_asymptotic_coefficient(k) for k in range(0, normcdfln_asymptotic_series_length)]
    printPolyFromRationals(normcdfln_asymptotic_variable_name, normcdfln_asymptotic_coefficients, normcdfln_asymptotic_indent)

if __name__ == '__main__': main()