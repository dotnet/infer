# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
#-----------------------------------------------------------------------------------
# Infer.NET IronPython example: Bayes Point Machine with 2 features
#-----------------------------------------------------------------------------------
import sys, clr, System, os
ruta00 = (os.environ['UserProfile'] + "\\.nuget\\packages\\microsoft.ml.probabilistic.compiler\\0.4.2203.202\\lib\\netstandard2.0")
ruta01 = (os.environ['UserProfile'] + "\\.nuget\\packages\\microsoft.ml.probabilistic\\0.4.2203.202\\lib\\netstandard2.0")

sys.path.append(ruta00)
sys.path.append(ruta01)

clr.AddReference("Microsoft.ML.Probabilistic")
clr.AddReference("Microsoft.ML.Probabilistic.Compiler")
import Microsoft.ML.Probabilistic.Compiler
from  Microsoft.ML.Probabilistic.Models import InferenceEngine
from System import Int32 as intG
import InferNetWrapper
from InferNetWrapper import *

def bayes_point_machine():
    print("\n\n------------------ Infer.NET Bayes Point Machine example ------------------\n");
    # The model
    len = Variable.New[intG]()
    j = Range(len)
    x = Variable.Array[Vector](j)
    y = Variable.Array[bool](j)
    w0 = VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3))
    w = Variable.Random[Vector](w0)
    noise = 0.1
    y[j] = Variable.GaussianFromMeanAndVariance(
        Variable.InnerProduct(w, x[j]).Named("innerProduct"), noise) > 0

    # The data
    incomes = System.Array[float]((63, 16, 28, 55, 22, 20 ))
    ages = System.Array[float]((38, 23, 40, 27, 18, 40 ))
    willBuy = System.Array[bool]((True, False, True, True, False, False))
    dataLen = willBuy.Length
    xdata = System.Array.CreateInstance(Vector, dataLen)
    for i in range(0, dataLen):
        xdata[i] = Vector.FromArray(System.Array[float]((incomes[i], ages[i], 1.0)))

    # Binding the data
    x.ObservedValue = xdata
    y.ObservedValue = willBuy
    len.ObservedValue = dataLen

    # Inferring the weights
    ie = InferenceEngine()
    wPosterior = ie.Infer[VectorGaussian](w)
    print ("Dist over w=\n", wPosterior)

    # Prediction
    incomesTest = System.Array[float]((58, 18, 22))
    agesTest = System.Array[float]((36, 24, 37))
    testDataLen = incomesTest.Length
    xtestData = System.Array.CreateInstance(Vector, testDataLen)
    for i in range(0, testDataLen):
        xtestData[i] = Vector.FromArray(System.Array[float]((incomesTest[i], agesTest[i], 1.0)))

    jtest = Range(testDataLen)
    xtest = Variable.Observed[Vector](xtestData, jtest)
    wtest = Variable.Random[Vector](wPosterior)
    ytest = Variable.Array[bool](jtest)
    ytest[jtest] = Variable.GaussianFromMeanAndVariance(
                      Variable.InnerProduct(wtest, xtest[jtest]), noise) > 0
    ypred = Distribution.ToArray[System.Array[Bernoulli]](ie.Infer(ytest))
    print( "Output = ", ypred[0], ",", ypred[1], ",", ypred[2])