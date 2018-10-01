# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
#-----------------------------------------------------------------------------------
# Infer.NET IronPython example: A mixture of 2 multivariate Gaussians
#-----------------------------------------------------------------------------------

import InferNetWrapper
from InferNetWrapper import *

# Initialisation function for breaking symmetry
def init_func() :
  return Discrete.PointMass(Rand.Int(2), 2)
  
# Generate data from function
def generate_data(n_data):
    trueM1 = Vector.FromArray(System.Array[float]((2.0, 3.0)))
    p1 = System.Array.CreateInstance(float,2,2); p1[0,0] = 3.0; p1[1,0] = 0.2; p1[0,1] = 0.2; p1[1,1] = 2.0
    trueP1 = PositiveDefiniteMatrix(p1)
    trueM2 = Vector.FromArray(System.Array[float]((7.0, 5.0)))
    p2 = System.Array.CreateInstance(float,2,2); p2[0,0] = 2.0; p2[1,0] = 0.4; p2[0,1] = 0.4; p2[1,1] = 4.0
    trueP2 = PositiveDefiniteMatrix(p2)
    trueVG1 = VectorGaussian.FromMeanAndPrecision(trueM1, trueP1)
    trueVG2 = VectorGaussian.FromMeanAndPrecision(trueM2, trueP2)
    truePi = 0.6
    trueB = Bernoulli(truePi)
    Rand.Restart(12347)
    data = System.Array.CreateInstance(Vector, n_data)
    for j in range(0, n_data):
        bSamp = trueB.Sample()
        if bSamp:
            data[j] = trueVG1.Sample()
        else:
           data[j] = trueVG2.Sample()
    return data

def gaussian_mixture(): 
    print("\n\n------------------ Infer.NET Mixture of Gaussians example ------------------\n");

    # Define a range for the number of mixture components
    k = Range(2)

    # Mixture component means
    means = Variable.Array[Vector](k).Named("means")
    mm0 = Vector.FromArray(0.0,0.0)
    mp0 = PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)
    means[k] = Variable.VectorGaussianFromMeanAndPrecision(mm0, mp0).ForEach(k)

    # Mixture component precisions
    precs = Variable.Array[PositiveDefiniteMatrix](k).Named("precs")
    precs[k] = Variable.WishartFromShapeAndScale(100.0, PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)).ForEach(k)

    # Mixture weights 
    weights = Variable.Dirichlet(k, System.Array[float]((1, 1))).Named("weights") 

    # Create a variable array which will hold the data
    n = Range(300)
    data = Variable.Array[Vector](n).Named("x")

    # Initialise to break symmetry
    length = n.SizeAsInt
    # Create latent indicator variable for each data point
    z = Variable.Array[int](n).Named("z")
    # Call initialiser from wrapper
    InitArray.init_var_arr(z, init_func, length)

    #mixture of gaussians
    with (Variable.ForEach(n)) :
         z[n] = Variable.Discrete(weights)
         with (Variable.Switch(z[n])) :
           data[n] = Variable.VectorGaussianFromMeanAndPrecision(means[z[n]], precs[z[n]])
          
    # Binding the data
    data.ObservedValue = generate_data(n.SizeAsInt)

    # The inference
    ie = InferenceEngine(VariationalMessagePassing())
    wDist = ie.Infer(weights)
    print "Estimated means for pi = (", wDist.GetMean().ToString(), ")"
    print "Distribution over pi = ", wDist.ToString()
    print "Distribution over vector Gaussian means ="
    print ie.Infer(means)
    print "Distribution over vector Gaussian precisions ="
    print ie.Infer(precs)
