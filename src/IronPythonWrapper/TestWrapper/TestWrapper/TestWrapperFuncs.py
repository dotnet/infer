# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
import InferNetWrapper
from InferNetWrapper import *

def TestFuncs():
  
    #-------------------------------------------------------------------------------------------------------
    #----------------INITIALISE FUNCTIONS-------------------------------------------------------------------
    k = Range(2)

    def init_init_func() :
      return Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt)
      
    def float_init_func():
      return Gaussian.FromMeanAndVariance(Rand.Normal(), 1)

    def bool_init_func():
      return Bernoulli.PointMass(True)
       
    def vector_init_func(): 
      return VectorGaussian.PointMass(Rand.Double())
       
    def pdm_init_func():
      return Wishart.PointMass(Rand.Double())
      
    #-------------------------------------------------------------------------------------------------------
    #test Wrapper Functions---------------------------------------------------------------------------------

    #----------------test 1D with int, float and bool types......
    # Create latent indicator variable for each data point
    length = 300
    n = Range(length)
   
    print "\n\n------------  Testing variable creation initialisation methods  -----------------\n"
    
    z1 = Variable.Array[int](n).Named("z1")
    InitArray.init_var_arr( z1, init_init_func, length)
    print "\nVariable array of ints:", type(z1)
    #
    f1 =  Variable.Array[float](n);
    InitArray.init_var_arr(f1, float_init_func, length)
    print "\nVariable array of bools:", type(f1)

    b1 = Variable.Array[bool](n);
    InitArray.init_var_arr(b1, bool_init_func, length)
    print "\nVariable array of bools:", type(b1)

    v1 = Variable.Array[Vector](n)
    InitArray.init_var_arr(v1, vector_init_func, length)
    print "\nVariable array of vectors:", type(v1)
    
    m1 = Variable.Array[PositiveDefiniteMatrix](n)
    InitArray.init_var_arr(m1, pdm_init_func, length)
    print "\nVariable array of PDMs:", type(m1)

    z2 = Variable.Array[Vector](n, n).Named("z2")
    print "\nVariable 2D array of vectors:", type(z2)
    InitArray.init_2D_var_arr( z2, vector_init_func, length,length)
    
    # Array of arrays
    jagged_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    z3 = CreateArray.create_var_arr_of_arr(jagged_lengths, float)
    print "\nVariable array of array of floats:", type(z3)
    InitArray.init_jagged_var_arr(z3, float_init_func, jagged_lengths)

    # 2D array of arrays
    jagged_2Dlengths = [[1, 2, 3], [4, 3, 2]]
    z4 = CreateArray.create_var_2D_arr_of_arr(jagged_2Dlengths, float)
    print "\nVariable 2D array of array of floats:", type(z4)
    InitArray.init_2D_jagged_var_arr(z4, float_init_func, jagged_2Dlengths)
    
    # Array of 2D arrays
    jagged_lengths2D = [ [1, 2], [3, 4], [5, 6], [4, 3], [4, 1] ]
    z5 = CreateArray.create_var_arr_of_2D_arr(jagged_lengths2D, float)
    print "\nVariable array of 2D array of floats:", type(z5)
    InitArray.init_jagged_2D_var_arr(z5, float_init_func, jagged_lengths2D)
     