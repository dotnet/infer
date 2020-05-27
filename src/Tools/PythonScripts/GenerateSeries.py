# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
from __future__ import division
from sympy import zeta, evalf, bernoulli, symbols, Poly, series, factorial, factorial2, S, log, exp, gamma, digamma, sqrt

# configuration
default_decimal_precision = 30
default_evalf_inner_precision = 500

gamma_at_2_series_length = 26
gamma_at_2_variable_name = "dx"
gamma_at_2_indent = "                    "

digamma_at_1_series_length = 2
digamma_at_1_variable_name = "x"
digamma_at_1_indent = "                    "

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

log_1_plus_series_length = 13
log_1_plus_variable_name = "x"
log_1_plus_indent = "                    "

log_1_minus_series_length = 11
log_1_minus_variable_name = "expx"
log_1_minus_indent = "                    "

x_minus_log_1_plus_series_length = 7
x_minus_log_1_plus_variable_name = "x"
x_minus_log_1_plus_indent = "                    "

exp_minus_1_series_length = 9
exp_minus_1_variable_name = "x"
exp_minus_1_indent = "                    "

exp_minus_1_ratio_minus_1_ratio_minus_half_series_length = 13
exp_minus_1_ratio_minus_1_ratio_minus_half_variable_name = "x"
exp_minus_1_ratio_minus_1_ratio_minus_half_indent = "                    "

log_exp_minus_1_ratio_series_length = 5
log_exp_minus_1_ratio_variable_name = "x"
log_exp_minus_1_ratio_indent = "                    "

normcdfln_asymptotic_series_length = 16
normcdfln_asymptotic_variable_name = "z"
normcdfln_asymptotic_indent = "                    "

one_minus_sqrt_one_minus_series_length = 5
one_minus_sqrt_one_minus_variable_name = "x"
one_minus_sqrt_one_minus_indent = "                    "

reciprocal_factorial_minus_1_series_length = 17
reciprocal_factorial_minus_1_variable_name = "x"
reciprocal_factorial_minus_1_indent = "                "

gamma_minus_reciprocal_series_length = 7
gamma_minus_reciprocal_variable_name = "x"
gamma_minus_reciprocal_indent = "                    "

def print_heading_comment(indent, header):
    print(f"{indent}// Truncated series {header}")
    print(f"{indent}// Generated automatically by /src/Tools/PythonScripts/GenerateSeries.py")

def format_real_coefficient(coefficient, decimal_precision = default_decimal_precision, evalf_inner_precision = default_evalf_inner_precision):
    return str(coefficient.evalf(decimal_precision, maxn=evalf_inner_precision))

def print_polynomial_with_real_coefficients(varname, coefficients, indent):
    if len(coefficients) <= 1:
        print(f"{indent}{format_real_coefficient(coefficients[0])}")
        return
    if coefficients[0] != 0.0:
        print(f"{indent}{format_real_coefficient(coefficients[0])} +")
    last_non_zero_idx = len(coefficients) - 1
    while coefficients[last_non_zero_idx] == 0.0:
        last_non_zero_idx = last_non_zero_idx - 1
    idx = 1
    parentheses = 0
    print(indent, end='')
    while idx < last_non_zero_idx:
        print(f"{varname} * ", end='')
        if coefficients[idx] != 0.0:
            print(f"({format_real_coefficient(coefficients[idx])} +")
            print(indent, end='')
            parentheses = parentheses + 1
        idx = idx + 1
    print(f"{varname} * {format_real_coefficient(coefficients[last_non_zero_idx])}", end='')
    for i in range(0, parentheses):
        print(")", end='')
    print()

def print_big_float_array(coefficients, decimal_precision, evalf_inner_precision):
    print("new BigFloat[]")
    print("{")
    last_non_zero_idx = len(coefficients) - 1
    while coefficients[last_non_zero_idx] == 0.0:
        last_non_zero_idx = last_non_zero_idx - 1
    idx = 0
    while idx < last_non_zero_idx:
        print(f'    BigFloatFactory.Create("{format_real_coefficient(coefficients[idx], decimal_precision, evalf_inner_precision)}"),')
        idx = idx + 1
    print(f'    BigFloatFactory.Create("{format_real_coefficient(coefficients[last_non_zero_idx], decimal_precision, evalf_inner_precision)}")')
    print("};")

def format_rational_coefficient(coefficient):
    return str(coefficient).replace("/", ".0 / ") + ".0"

def print_polynomial_with_rational_coefficients(varname, coefficients, indent):
    if len(coefficients) <= 1:
        print(f"{indent}{format_rational_coefficient(coefficients[0])}")
        return
    if coefficients[0] != 0:
        print(f"{indent}{format_rational_coefficient(coefficients[0])} +")
    last_non_zero_idx = len(coefficients) - 1
    while coefficients[last_non_zero_idx] == 0:
        last_non_zero_idx = last_non_zero_idx - 1
    idx = 1
    parentheses = 0
    print(indent, end='')
    while idx <= last_non_zero_idx:
        print(f"{varname} * ", end='')
        if coefficients[idx] != 0:
            if idx < last_non_zero_idx:
                suffix = ' +'
            else:
                suffix = ''
            print(f"({format_rational_coefficient(coefficients[idx])}{suffix}")
            print(indent, end='')
            parentheses = parentheses + 1
        idx = idx + 1
    for i in range(0, parentheses):
        print(")", end='')
    print()

def gamma_at_2_coefficient(k):
    if k == 0:
        return S(0)
    return ((-1)**(k + 1)*(zeta(k + 1) - 1)/(k + 1))

def digamma_at_1_coefficient(k):
    if k == 0:
        return digamma(1)
    return ((-1)**(k + 1) * zeta(k + 1))

def digamma_at_2_coefficient(k):
    if k == 0:
        return S(0)
    return ((-1)**(k + 1)*(zeta(k + 1) - 1))

def digamma_asymptotic_coefficient(k):
    if k == 0:
        return S(0)
    return bernoulli(2 * k) / (2 * k)

def trigamma_at_1_coefficient(k):
    return ((-1)**k * (k + 1) * zeta(k + 2))

def trigamma_asymptotic_coefficient(k):
    if k == 0:
        return S(0)
    return bernoulli(2 * k)

def tetragamma_at_1_coefficient(k):
    return ((-1)**(k + 1) * (k + 1) * (k + 2) * zeta(k + 3))

def tetragamma_asymptotic_coefficient(k):
    if k == 0:
        return S(0)
    return -(2 * k - 1) * bernoulli(2 * k - 2)

def gammaln_asymptotic_coefficient(k):
    return bernoulli(2 * k + 2) / (2 * (k + 1) * (2 * k + 1))

def log_1_plus_coefficient(k):
    if k == S(0):
        return S(0)
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
        return S(0)
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

def get_one_minus_sqrt_one_minus_coefficients(count):
    x = symbols('x')
    return list(reversed(Poly((1 - sqrt(1 - x)).series(x, 0, count).removeO()).all_coeffs()))

def get_reciprocal_factorial_minus_1_coefficients(count):
    x = symbols('x')
    return list(reversed(Poly((1 / gamma(x + 1) - 1).series(x, 0, count).removeO(), x).all_coeffs()))

def get_gamma_minus_reciprocal_coefficients(count):
    x = symbols('x')
    return list(reversed(Poly((gamma(x) - 1 / x).series(x, 0, count).removeO(), x).all_coeffs()))

def main():
    print_heading_comment(gamma_at_2_indent, "1: Gamma at 2")
    gamma_at_2_coefficients = [gamma_at_2_coefficient(k) for k in range(0, gamma_at_2_series_length)]
    print_polynomial_with_real_coefficients(gamma_at_2_variable_name, gamma_at_2_coefficients, gamma_at_2_indent)

    print_heading_comment(digamma_at_1_indent, "2: Digamma at 1")
    digamma_at_1_coefficients = [digamma_at_1_coefficient(k) for k in range(0, digamma_at_1_series_length)]
    print_polynomial_with_real_coefficients(digamma_at_1_variable_name, digamma_at_1_coefficients, digamma_at_1_indent)

    print_heading_comment(digamma_at_2_indent, "3: Digamma at 2")
    digamma_at_2_coefficients = [digamma_at_2_coefficient(k) for k in range(0, digamma_at_2_series_length)]
    print_polynomial_with_real_coefficients(digamma_at_2_variable_name, digamma_at_2_coefficients, digamma_at_2_indent)

    print_heading_comment(digamma_asymptotic_indent, "4: Digamma asymptotic")
    digamma_asymptotic_coefficients = [digamma_asymptotic_coefficient(k) for k in range(0, digamma_asymptotic_series_length)]
    print_polynomial_with_rational_coefficients(digamma_asymptotic_variable_name, digamma_asymptotic_coefficients, digamma_asymptotic_indent)

    print_heading_comment(trigamma_at_1_indent, "5: Trigamma at 1")
    trigamma_at_1_coefficients = [trigamma_at_1_coefficient(k) for k in range(0, trigamma_at_1_series_length)]
    print_polynomial_with_real_coefficients(trigamma_at_1_variable_name, trigamma_at_1_coefficients, trigamma_at_1_indent)

    print_heading_comment(trigamma_asymptotic_indent, "6: Trigamma asymptotic")
    trigamma_asymptotic_coefficients = [trigamma_asymptotic_coefficient(k) for k in range(0, trigamma_asymptotic_series_length)]
    print_polynomial_with_rational_coefficients(trigamma_asymptotic_variable_name, trigamma_asymptotic_coefficients, trigamma_asymptotic_indent)

    print_heading_comment(tetragamma_at_1_indent, "7: Tetragamma at 1")
    tetragamma_at_1_coefficients = [tetragamma_at_1_coefficient(k) for k in range(0, tetragamma_at_1_series_length)]
    print_polynomial_with_real_coefficients(tetragamma_at_1_variable_name, tetragamma_at_1_coefficients, tetragamma_at_1_indent)

    print_heading_comment(tetragamma_asymptotic_indent, "8: Tetragamma asymptotic")
    tetragamma_asymptotic_coefficients = [tetragamma_asymptotic_coefficient(k) for k in range(0, tetragamma_asymptotic_series_length)]
    print_polynomial_with_rational_coefficients(tetragamma_asymptotic_variable_name, tetragamma_asymptotic_coefficients, tetragamma_asymptotic_indent)

    print_heading_comment(gammaln_asymptotic_indent, "9: GammaLn asymptotic")
    gammaln_asymptotic_coefficients = [gammaln_asymptotic_coefficient(k) for k in range(0, gammaln_asymptotic_series_length)]
    print_polynomial_with_rational_coefficients(gammaln_asymptotic_variable_name, gammaln_asymptotic_coefficients, gammaln_asymptotic_indent)

    print_heading_comment(log_1_plus_indent, "10: log(1 + x)")
    log_1_plus_coefficients = [log_1_plus_coefficient(k) for k in range(0, log_1_plus_series_length)]
    print_polynomial_with_rational_coefficients(log_1_plus_variable_name, log_1_plus_coefficients, log_1_plus_indent)

    print_heading_comment(log_1_minus_indent, "11: log(1 - x)")
    log_1_minus_coefficients = [log_1_minus_coefficient(k) for k in range(0, log_1_minus_series_length)]
    print_polynomial_with_rational_coefficients(log_1_minus_variable_name, log_1_minus_coefficients, log_1_minus_indent)

    print_heading_comment(x_minus_log_1_plus_indent, "12: x - log(1 + x)")
    x_minus_log_1_plus_coefficients = [x_minus_log_1_plus_coefficient(k) for k in range(0, x_minus_log_1_plus_series_length)]
    print_polynomial_with_rational_coefficients(x_minus_log_1_plus_variable_name, x_minus_log_1_plus_coefficients, x_minus_log_1_plus_indent)

    print_heading_comment(exp_minus_1_indent, "13: exp(x) - 1")
    exp_minus_1_coefficients = [exp_minus_1_coefficient(k) for k in range(0, exp_minus_1_series_length)]
    print_polynomial_with_rational_coefficients(exp_minus_1_variable_name, exp_minus_1_coefficients, exp_minus_1_indent)

    print_heading_comment(exp_minus_1_ratio_minus_1_ratio_minus_half_indent, "14: ((exp(x) - 1) / x - 1) / x - 0.5")
    exp_minus_1_ratio_minus_1_ratio_minus_half_coefficients = [exp_minus_1_ratio_minus_1_ratio_minus_half_coefficient(k) for k in range(0, exp_minus_1_ratio_minus_1_ratio_minus_half_series_length)]
    print_polynomial_with_rational_coefficients(exp_minus_1_ratio_minus_1_ratio_minus_half_variable_name, exp_minus_1_ratio_minus_1_ratio_minus_half_coefficients, exp_minus_1_ratio_minus_1_ratio_minus_half_indent)

    print_heading_comment(log_exp_minus_1_ratio_indent, "15: log(exp(x) - 1) / x")
    log_exp_minus_1_ratio_coefficients = get_log_exp_minus_1_ratio_coefficients(log_exp_minus_1_ratio_series_length)
    print_polynomial_with_rational_coefficients(log_exp_minus_1_ratio_variable_name, log_exp_minus_1_ratio_coefficients, log_exp_minus_1_ratio_indent)

    print_heading_comment(normcdfln_asymptotic_indent, "16: normcdfln asymptotic")
    normcdfln_asymptotic_coefficients = [normcdfln_asymptotic_coefficient(k) for k in range(0, normcdfln_asymptotic_series_length)]
    print_polynomial_with_rational_coefficients(normcdfln_asymptotic_variable_name, normcdfln_asymptotic_coefficients, normcdfln_asymptotic_indent)
    
    print_heading_comment(one_minus_sqrt_one_minus_indent, "17: 1 - sqrt(1 - x)")
    one_minus_sqrt_one_minus_coefficients = get_one_minus_sqrt_one_minus_coefficients(one_minus_sqrt_one_minus_series_length)
    print_polynomial_with_rational_coefficients(one_minus_sqrt_one_minus_variable_name, one_minus_sqrt_one_minus_coefficients, one_minus_sqrt_one_minus_indent)
    
    print_heading_comment(reciprocal_factorial_minus_1_indent, "18: Reciprocal factorial minus 1")
    reciprocal_factorial_minus_1_coefficients = get_reciprocal_factorial_minus_1_coefficients(reciprocal_factorial_minus_1_series_length)
    print_polynomial_with_real_coefficients(reciprocal_factorial_minus_1_variable_name, reciprocal_factorial_minus_1_coefficients, reciprocal_factorial_minus_1_indent)
    
    print_heading_comment(gamma_minus_reciprocal_indent, "19: Gamma(x) - 1/x")
    gamma_minus_reciprocal_coefficients = get_gamma_minus_reciprocal_coefficients(gamma_minus_reciprocal_series_length)
    print_polynomial_with_real_coefficients(gamma_minus_reciprocal_variable_name, gamma_minus_reciprocal_coefficients, gamma_minus_reciprocal_indent)

def big_float_main():
    print_heading_comment(trigamma_at_1_indent, "5: Trigamma at 1")
    trigamma_at_1_coefficients = [trigamma_at_1_coefficient(k) for k in range(0, 10)]
    print_big_float_array(trigamma_at_1_coefficients, 50, 500)

    print_heading_comment(trigamma_asymptotic_indent, "6: Trigamma asymptotic")
    trigamma_asymptotic_coefficients = [trigamma_asymptotic_coefficient(k) for k in range(0, 32)]
    print_big_float_array(trigamma_asymptotic_coefficients, 50, 500)

    print_heading_comment(tetragamma_at_1_indent, "7: Tetragamma at 1")
    tetragamma_at_1_coefficients = [tetragamma_at_1_coefficient(k) for k in range(0, 11)]
    print_big_float_array(tetragamma_at_1_coefficients, 50, 500)

    print_heading_comment(tetragamma_asymptotic_indent, "8: Tetragamma asymptotic")
    tetragamma_asymptotic_coefficients = [tetragamma_asymptotic_coefficient(k) for k in range(0, 32)]
    print_big_float_array(tetragamma_asymptotic_coefficients, 50, 500)

    print_heading_comment(gammaln_asymptotic_indent, "9: GammaLn asymptotic")
    gammaln_asymptotic_coefficients = [gammaln_asymptotic_coefficient(k) for k in range(0, 31)]
    print_big_float_array(gammaln_asymptotic_coefficients, 50, 500)

    print_heading_comment(log_1_minus_indent, "11: log(1 - x)")
    log_1_minus_coefficients = [log_1_minus_coefficient(k) for k in range(0, 50)]
    print_big_float_array(log_1_minus_coefficients, 50, 500)

    print_heading_comment(x_minus_log_1_plus_indent, "12: x - log(1 + x)")
    x_minus_log_1_plus_coefficients = [x_minus_log_1_plus_coefficient(k) for k in range(0, 26)]
    print_big_float_array(x_minus_log_1_plus_coefficients, 50, 500)

    print_heading_comment(exp_minus_1_ratio_minus_1_ratio_minus_half_indent, "14: ((exp(x) - 1) / x - 1) / x - 0.5")
    exp_minus_1_ratio_minus_1_ratio_minus_half_coefficients = [exp_minus_1_ratio_minus_1_ratio_minus_half_coefficient(k) for k in range(0, 19)]
    print_big_float_array(exp_minus_1_ratio_minus_1_ratio_minus_half_coefficients, 50, 500)

    print_heading_comment(normcdfln_asymptotic_indent, "16: normcdfln asymptotic")
    normcdfln_asymptotic_coefficients = [normcdfln_asymptotic_coefficient(k) for k in range(0, 19)]
    print_big_float_array(normcdfln_asymptotic_coefficients, 50, 500)
    
    print_heading_comment(reciprocal_factorial_minus_1_indent, "18: Reciprocal factorial minus 1")
    reciprocal_factorial_minus_1_coefficients = get_reciprocal_factorial_minus_1_coefficients(22)
    print_big_float_array(reciprocal_factorial_minus_1_coefficients, 50, 500)
    
    print_heading_comment(gamma_minus_reciprocal_indent, "19: Gamma(x) - 1/x")
    gamma_minus_reciprocal_coefficients = get_gamma_minus_reciprocal_coefficients(30)
    print_big_float_array(gamma_minus_reciprocal_coefficients, 50, 500)

if __name__ == '__main__': big_float_main()