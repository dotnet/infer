import types
import numpy as np
from random import random

import clr
from System import Boolean, Double

clr.AddReference("Microsoft.ML.Probabilistic")
clr.AddReference("Microsoft.ML.Probabilistic.Compiler")
from Microsoft.ML.Probabilistic import *
from Microsoft.ML.Probabilistic.Distributions import VectorGaussian
from Microsoft.ML.Probabilistic.Math import Vector, DenseVector, PositiveDefiniteMatrix
from Microsoft.ML.Probabilistic.Models import Variable, InferenceEngine, Range, VariableArray

from System import Array

import Microsoft.ML.Probabilistic.Math


def __list_methods(obj):
    for m in dir(obj):
        try:
            print(m)
        except:
            pass

    quit()


def __list_method_overloads(method):
    for m in f'{method.__overloads__}'.split('\n'):
        print(' '.join(m.split(' ')[1:]))

    quit()


def sym1():
    # TODO: no conversion int -> double
    theta1 = Variable.GaussianFromMeanAndVariance(0.0, 100.0)
    theta1.Name = 'theta1'

    theta2 = Variable.GaussianFromMeanAndVariance(0.0, 100.0)
    theta2.Name = 'theta2'

    # TODO: operators not implemented
    observedThetaSum = theta1.op_Addition(theta1, theta2)
    observedThetaSum.ObservedValue = 2.0

    engine = InferenceEngine()

    t1 = engine.Infer(theta1)

    print(t1)


def sym2():
    prior_variance = 100.0
    noise_variance = 1.0

    m = Variable.GaussianFromMeanAndVariance(0.0, 100.0).Named('m')
    y = Variable.GaussianFromMeanAndVariance(m, noise_variance).Named("y")
    y.ObservedValue = 0.0

    engine = InferenceEngine()

    t = engine.Infer(m)
    print(t)


def sym_hierarchy():
    prior_variance = 100.0
    noise_variance = 1.0

    groups = 3
    observations = 1

    b = Variable.GaussianFromMeanAndVariance(0.0, prior_variance).Named("b")
    c = []

    for i in range(groups):
        c.append(Variable.GaussianFromMeanAndVariance(0.0, prior_variance).Named(f"c{i + 1}"))

    for i in range(groups):
        for j in range(observations):
            y = Variable.GaussianFromMeanAndVariance(b.op_Addition(b, c[i]), noise_variance).Named(f"y{i}{j}")
            y.ObservedValue = i - 1

    engine = InferenceEngine()
    print(engine.Infer(b))


def learn_gauss():
    size = 1000
    data = [d for d in np.random.normal(0, 1, size=size)]

    mean = Variable.GaussianFromMeanAndVariance(0.0, 1.0).Named("mean")
    precision = Variable.GammaFromShapeAndScale(1.0, 1.0).Named("precision")

    for i in range(len(data)):
        x = Variable.GaussianFromMeanAndPrecision(mean, precision)
        x.ObservedValue = data[i]

    engine = InferenceEngine()
    print(engine.Infer(mean))
    print(engine.Infer(precision))


def learn_gauss_range():
    size = 1000
    data = [d for d in np.random.normal(0, 1, size=size)]

    mean = Variable.GaussianFromMeanAndVariance(0.0, 1.0).Named("mean")
    precision = Variable.GammaFromShapeAndScale(1.0, 1.0).Named("precision")

    data_range = Range(len(data)).Named("n")

    x = Variable.Array[Double](data_range)
    v = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(data_range)

    # TODO: setter types not inferred
    x.set_Item.Overloads[Range, Variable[Double]](data_range, v)

    x.ObservedValue = data

    engine = InferenceEngine()
    print(engine.Infer(mean))
    print(engine.Infer(precision))


def bayes_point_machine():
    will_buy = [True, False, True, True, False, False]

    # TODO: Cannot create arrays of TypedReference, ArgIterator, ByRef... at Python.Runtime.Converter.ToArray
    y_data = Array[bool](will_buy)

    # TODO: generic + overload does not work?
    # y = Variable.Observed[bool].Overloads[Array[bool]](ydata)
    y_data_range = Range(len(y_data)).Named("y_range")

    y = Variable.Array[bool](y_data_range)
    y.ObservedValue = y_data

    incomes = [63, 16, 28, 55, 22, 20]
    ages = [38, 23, 40, 27, 18, 40]

    # xdata = [Vector.FromArray(income, age, 1) for income, age in zip(incomes, ages)]
    x_data = Array[Vector]([Vector.FromArray(income, age, 1) for income, age in zip(incomes, ages)])
    # x_data_range = Range(len(x_data)).Named("x_range")

    # x = Variable.Observed(xdata)
    x = Variable.Array[Vector](y_data_range)
    x.ObservedValue = x_data

    # BPM
    eye = PositiveDefiniteMatrix.Identity(3)
    m = Vector.Zero(3)
    vg = VectorGaussian.Overloads[Vector, PositiveDefiniteMatrix](m, eye)
    w = Variable.Random[Vector](vg)

    noise = 0.1
    j = y.Range
    # TODO: again not inferred
    # y[j] = Variable.GaussianFromMeanAndVariance(Variable.InnerProduct(w, x[j]), noise) > 0
    ip = Variable.InnerProduct(w, x.get_Item.Overloads[Range](j))
    v = Variable.GaussianFromMeanAndVariance(ip, noise)
    v = v.op_GreaterThan(v, 0.0)
    y.set_Item.Overloads[Range, Variable[Boolean]](j, v)

    engine = InferenceEngine()
    w_posterior = engine.Infer(w)
    print(w_posterior)


def clinical_trial():
    # A healthy challenge

    control_group_data = Array[bool]([False, False, True, False, False])
    control_group = Variable.Array[bool](Range(len(control_group_data)))
    control_group.ObservedValue = control_group_data

    treated_group_data = Array[bool]([True, False, True, True, True])
    treated_group = Variable.Array[bool](Range(len(treated_group_data)))
    treated_group.ObservedValue = treated_group_data

    i = control_group.Range
    j = treated_group.Range

    # Prior on being effective treatment
    is_effective = Variable.Bernoulli(0.5)

    # Cause and effect

    # if block
    if_var = Variable.If(is_effective)
    prob_if_control = Variable.Beta(1.0, 1.0)
    t = Variable.Bernoulli(prob_if_control).ForEach(i)
    control_group.set_Item.Overloads[Range, Variable[Boolean]](i, t)

    prob_if_treated = Variable.Beta(1.0, 1.0)
    t = Variable.Bernoulli(prob_if_treated).ForEach(j)
    treated_group.set_Item.Overloads[Range, Variable[Boolean]](j, t)
    if_var.Dispose()

    # A bit of background
    if_var = Variable.IfNot(is_effective)
    prob_all = Variable.Beta(1.0, 1.0)
    control_group.set_Item.Overloads[Range, Variable[Boolean]](i, Variable.Bernoulli(prob_all).ForEach(i))
    treated_group.set_Item.Overloads[Range, Variable[Boolean]](j, Variable.Bernoulli(prob_all).ForEach(j))
    if_var.Dispose()

    # Clinical accuracy

    engine = InferenceEngine()
    print(engine.Infer(is_effective))


clinical_trial()
