# Licensed to the .NET Foundation under one or more agreements.
# The .NET Foundation licenses this file to you under the MIT license.
# See the LICENSE file in the project root for more information.
import System
from System import Console
from System import Array
from System import IO
import clr
import sys

#add path to infer.net dlls .................................
#sys.path.append(r"C:\Program Files\Microsoft Research\Infer.NET 2.4\bin\Release")
clr.AddReferenceToFile("Microsoft.ML.Probabilistic.Compiler.dll")
clr.AddReferenceToFile("Microsoft.ML.Probabilistic.dll")

#---import all namespaces----------------
import Microsoft.ML.Probabilistic
import Microsoft.ML.Probabilistic.Models
import Microsoft.ML.Probabilistic.Distributions
import Microsoft.ML.Probabilistic.Collections
import Microsoft.ML.Probabilistic.Factors
import Microsoft.ML.Probabilistic.Math
import Microsoft.ML.Probabilistic.Algorithms
from Microsoft.ML.Probabilistic import * 

#---------import all classes and methods from above namespaces-----------
from Microsoft.ML.Probabilistic.Distributions import *
from Microsoft.ML.Probabilistic.Models import *
from Microsoft.ML.Probabilistic.Collections import *
from Microsoft.ML.Probabilistic.Factors import *
from Microsoft.ML.Probabilistic.Math import *
from Microsoft.ML.Probabilistic.Algorithms import *

InferenceEngine.DefaultEngine.Compiler.CompilerChoice = Microsoft.ML.Probabilistic.Compiler.CompilerChoice.Roslyn

#----------------------------------------------------------------------------
#----------------  Variable array helper methos   ---------------------------
#----------------------------------------------------------------------------
class CreateArray : 
    def __init__(self):
        pass
    #------------------------------------------------------------------------
    #  Creates a Variable which is a 1-D array of 1-D arrays 
    #------------------------------------------------------------------------
    @classmethod
    def create_var_arr_of_arr(cls, jagged_lengths, type):
       
        length = len(jagged_lengths)
        # create 1D .NET array of lengths from Python list
        sizes = System.Array.CreateInstance(int, length)
        for i in range(0, length):
            sizes[i] = jagged_lengths[i] 
        outer = Range(length)
        # make a constant VariableArray from the length array
        sizes_var = Variable.Constant[int](sizes, outer)
        # create the inner range (which varies with the outer range)
        inner = Range(sizes_var[outer])
        # create and return the Variable
        return Variable.Array[type](Variable.Array[type](inner), outer)
            
    #------------------------------------------------------------------------
    #  Creates a Variable which is a 2-D array of 1-D arrays 
    #------------------------------------------------------------------------
    @classmethod
    def create_var_2D_arr_of_arr(cls, jagged_2D_lengths, type):
       
        length1 = len(jagged_2D_lengths)
        length2 = len(jagged_2D_lengths[0])
        # create 2D .NET array of lengths from Python list or lists
        len_array = System.Array.CreateInstance(int, length1, length2)
        for i in range(0, length1):
            for j in range(0, length2):
                len_array[i, j] = jagged_2D_lengths[i][j]
        outer1 = Range(length1)
        outer2 = Range(length2)
        # make a constant 2D VariableArray from the 2D length array
        sizes_var = Variable.Constant[int](len_array, outer1, outer2)
        # create the inner range (which varies with the outer ranges)
        inner = Range(sizes_var[outer1, outer2])
        # create and return the Variable
        return Variable.Array[type](Variable.Array[type](inner), outer1, outer2)
        
    #------------------------------------------------------------------------
    #  Creates a Variable which is a 1-D array of 2-D arrays 
    #------------------------------------------------------------------------
    @classmethod
    def create_var_arr_of_2D_arr(cls, jagged_lengths2D, type):
       
        length = len(jagged_lengths2D)
        outer = Range(length)
        sizes1 = System.Array.CreateInstance(int, length)
        for i in range(0, length):
            sizes1[i] = jagged_lengths2D[i][0] 
        sizes2 = System.Array.CreateInstance(int, length)
        for i in range(0, length):
            sizes2[i] = jagged_lengths2D[i][1] 
        # make constant VariableArrays from the length arrays
        sizes1_var = Variable.Constant[int](sizes1, outer)
        sizes2_var = Variable.Constant[int](sizes2, outer)
        # create the inner ranges (which vary with the outer range)
        inner1 = Range(sizes1_var[outer])
        inner2 = Range(sizes2_var[outer])
        return Variable.Array[type](Variable.Array[type](inner1, inner2), outer)

#----------------------------------------------------------------------------
#----------------- Initialisation helper methods ----------------------------
#----------------------------------------------------------------------------
class InitArray : 

    def __init__(self):
        pass   

    #helper function for type conversions
    @classmethod
    def _upcast_type(cls, clrTypeName):
       if clrTypeName == Microsoft.ML.Probabilistic.Math.DenseVector:
            return Vector
       elif clrTypeName == Microsoft.ML.Probabilistic.Math.SparseVector:
            return Vector
       else :
           return clrTypeName

    #----------------------------------------------------
    #  1D case
    #----------------------------------------------------
    @classmethod
    def _create_arr_of_dist(cls, length, init_func):
        
        if callable(init_func):
          t = type(init_func())
        cinit = System.Array.CreateInstance(t, length) 
        for i in range(0,length):
          cinit[i] = init_func()
        return cinit 
       
    @classmethod
    def init_var_arr_from_data(cls, init_var, dist_array):
       
       distType = dist_array.GetType().GetElementType()
       baseType = clr.GetPythonType(Distribution.GetDomainType(dist_array[0].GetType()))
       bt = InitArray._upcast_type(baseType)
       da = Distribution[bt].Array[distType](dist_array)
       datype = type(da)
       init_var.InitialiseTo[datype](da)
       
    @classmethod
    def init_var_arr(cls, vArray, init_func, length):
        
       initSystemArray = InitArray._create_arr_of_dist(length, init_func)
       return InitArray.init_var_arr_from_data(vArray, initSystemArray)    
        
    #----------------------------------------------------
    #  Non-jagged 2D case
    #----------------------------------------------------
    @classmethod
    def _create_2D_arr_of_dist(cls, length1, length2, init_func):
       
       if callable(init_func):
            t = type(init_func())
       result = System.Array.CreateInstance(t, length1, length2) 
       for i in range(0, length1):
          for j in range(0, length2):
              result[i,j] = init_func()
       return result

    @classmethod
    def init_2D_var_arr_from_data(cls, init_var, dist_array):
       
       distType = dist_array.GetType().GetElementType()
       baseType = clr.GetPythonType(Distribution.GetDomainType(dist_array[0,0].GetType()))
       bt = InitArray._upcast_type(baseType)
       da = Distribution[bt].Array[distType](dist_array)
       datype = type(da)
       init_var.InitialiseTo[datype](da)
       
    @classmethod
    def init_2D_var_arr(cls, vArray, init_func, length1, length2):
       
       initSystemArray = InitArray._create_2D_arr_of_dist(length1, length2, init_func)
       return InitArray.init_2D_var_arr_from_data(vArray, initSystemArray)   
       
    #----------------------------------------------------
    #  Array of array
    #----------------------------------------------------
    @classmethod
    def _create_jagged_arr_of_dist(cls, jagged_lengths, init_func):
        
        if callable(init_func):
           distType = type(init_func())
        length = len(jagged_lengths)
        result = System.Array.CreateInstance(System.Array[distType], length)
        for i in range(0, length):
            jaggedSize = jagged_lengths[i]
            result[i] =  System.Array.CreateInstance(distType, jaggedSize)
            for j in range(0,jaggedSize):
                result[i][j] = init_func()             
        return result

    @classmethod
    def init_jagged_var_arr_from_data(cls, init_var, dist_array):
        
        length1 = dist_array.Length
        #look for 1st non empty value and get distribution type etc
        isFound = False
        for i in range(0,length1):
          if isFound:break
          length2 = dist_array[i].Length
          for j in range(0,length2):
              if (dist_array[i][j] != None):
                  d = dist_array[i][j]
                  distType = type(d)
                  isFound = True
                  break
        baseType = clr.GetPythonType(Distribution.GetDomainType(d.GetType()))
        bt = InitArray._upcast_type(baseType)
        da = Distribution[bt].Array[distType](dist_array)
        datype = type(da)
        init_var.InitialiseTo[datype](da)

    @classmethod
    def init_jagged_var_arr(cls, vArray, init_func, jagged_lengths):
       length = len(jagged_lengths)
       initSystemArray = InitArray._create_jagged_arr_of_dist(jagged_lengths,init_func)
       return InitArray.init_jagged_var_arr_from_data(vArray, initSystemArray)   

    #----------------------------------------------------
    #  2D Jagged case
    #----------------------------------------------------

    @classmethod
    def _create_2D_jagged_arr_of_dist(cls, jagged_lengths, init_func):
        
        if callable(init_func):
           distType = type(init_func())
        length1 = len(jagged_lengths)
        length2 = len(jagged_lengths[0])
        result = System.Array.CreateInstance(System.Array[distType], length1, length2)
        for i in range(0,length1):
           for k in range(0,length2):
              result[i,k] =  System.Array.CreateInstance(distType, jagged_lengths[i][k])
              for j in range(0,jagged_lengths[i][k]):
                    result[i,k][j] = init_func()            
        return result
        
    @classmethod
    def init_2D_jagged_var_arr_from_data(cls, init_var, dist_array):
        
        length1 = dist_array.GetLength(0)
        length2 = dist_array.GetLength(1)
        isFound = False
        for i in range(0,length1):
          if isFound:break
          for j in range(0,length2):
             if isFound:break
             length3 = dist_array[i,j].Length
             for k in range(0,length3):
                if (dist_array[i,j][k] != None):
                   d = dist_array[i,j][k]
                   distType = type(d)
                   isFound = True
                   break
        baseType = clr.GetPythonType(Distribution.GetDomainType(d.GetType()))
        bt = InitArray._upcast_type(baseType)
        da = Distribution[bt].Array[distType](dist_array)
        datype = type(da)
        init_var.InitialiseTo[datype](da)
        
    @classmethod
    def init_2D_jagged_var_arr(cls, vArray, init_func, jagged_lengths):
        
        length1 = len(jagged_lengths)
        length2 = len(jagged_lengths[0])
        initSystemArray = InitArray._create_2D_jagged_arr_of_dist(jagged_lengths, init_func)
        return InitArray.init_2D_jagged_var_arr_from_data(vArray, initSystemArray)   
        
    #----------------------------------------------------
    #  Jagged 2D case
    #----------------------------------------------------
    @classmethod
    def _create_jagged_2D_arr_of_dist(cls, jagged_lengths, init_func):
        
        distType = type(init_func())
        length = len(jagged_lengths)
        t = clr.GetPythonType(System.Type.MakeArrayType(distType, 2))
        result = System.Array.CreateInstance(t, length)
        for i in range(0, length):
              l1 =  jagged_lengths[i][0]
              l2 =  jagged_lengths[i][1]
              result[i] =  System.Array.CreateInstance(distType, l1, l2)         
              for j in range(0, l1):
                   for k in range(0, l2):
                      result[i][j,k] = init_func()                                            
        return result

    @classmethod
    def init_jagged_2D_var_arr_from_data(cls, init_var, dist_array):
        
        length1 = dist_array.Length
        #find first non-empty element
        isFound = False
        for i in range(0, length1):
          if isFound:break
          length2 = dist_array[i].GetLength(0)
          length3 = dist_array[i].GetLength(1) 
          for j in range(0,length2):
             if (isFound): break
             for k in range(0,length3):
               if (isFound): break
               if (dist_array[i][j,k] != None) :
                   d = dist_array[i][j,k]
                   distType = type(d)
                   isFound = True
                   break
        baseType = clr.GetPythonType(Distribution.GetDomainType(d.GetType()))
        bt = InitArray._upcast_type(baseType)
        da = Distribution[bt].Array[distType](dist_array)
        datype = type(da)
        init_var.InitialiseTo[datype](da)     
             
    @classmethod
    def init_jagged_2D_var_arr(cls, vArray, init_func, jagged_lengths):
        
        length = len(jagged_lengths)
        initSystemArray = InitArray._create_jagged_2D_arr_of_dist(jagged_lengths, init_func)
        return InitArray.init_jagged_2D_var_arr_from_data(vArray, initSystemArray)
