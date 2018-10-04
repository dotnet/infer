# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
#-----------------------------------------------------------------------------------
# Infer.NET IronPython example: Learning a Gaussian
#-----------------------------------------------------------------------------------

import InferNetWrapper
from InferNetWrapper import *

def gaussian_ranges():
    print("\n\n------------------ Infer.NET Learning a Gaussian example ------------------\n");

    # The model
    len = Variable.New[int]()
    dataRange = Range(len)
    x = Variable.Array[float](dataRange)
    mean = Variable.GaussianFromMeanAndVariance(0, 100)
    precision = Variable.GammaFromShapeAndScale(1, 1)
    x[dataRange] = Variable.GaussianFromMeanAndPrecision(mean, precision).ForEach(dataRange)

    # The data
    data = System.Array.CreateInstance(float, 100)
    for i in range(0,100):
        data[i] = Rand.Normal(0, 1)

    # Binding the data
    len.ObservedValue = 100
    x.ObservedValue = data

    # The inference
    ie = InferenceEngine(Algorithms.VariationalMessagePassing())
    print "mean = ", ie.Infer(mean)
    print "prec = ", ie.Infer(precision)