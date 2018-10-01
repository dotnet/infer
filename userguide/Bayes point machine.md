---
layout: default 
--- 
[Infer.NET user guide](index.md) : [FSharp Wrapper](FSharp Wrapper.md)

## Bayes point machine tutorial in F\#

For a description of this tutorial and the C# code please see [the Bayes point machine tutorial](Bayes Point Machine tutorial.md).

#### F# script

```fsharp
#light // Reference the Infer.NET DLLs  
#r @"C:/Program Files/Microsoft Research/Infer.NET 2.2/bin/Release/Microsoft.ML.Probabilistic.Compiler.dll"  
#r @"C:/Program Files/Microsoft Research/Infer.NET 2.2/bin/Release/Microsoft.ML.Probabilistic.dll"  
#r @"C:/Program Files/Microsoft Research/Infer.NET 2.2/bin/Release/Microsoft.ML.Probabilistic.FSharp.dll"  

open Microsoft.ML.Probabilistic  
open Microsoft.ML.Probabilistic.Models  
open Microsoft.ML.Probabilistic.Distributions  
open Microsoft.ML.Probabilistic.Factors  
open Microsoft.ML.Probabilistic.Math  
open Microsoft.ML.Probabilistic.FSharp  

//-----------------------------------------------------------------------------------  
// Infer.NET: F# script for a Bayes Point Machine with 2 features  
//-----------------------------------------------------------------------------------  

// The training model  
let noise = 0.1let len = Variable.New<int>()  
let j = Range(len)  
let x = Variable.Array<Vector>(j)  
let w0 = VectorGaussian(Vector.Zero(3), PositiveDefiniteMatrix.Identity(3))  
let w = Variable.Random<Vector>(w0)  
let y = Variable.AssignVariableArray  
            (Variable.Array<bool>(j)) j  
            (fun j -> Variable.IsPositive(Variable.GaussianFromMeanAndVariance  
                          (Variable.InnerProduct(w, x.[j]), noise)))  

// The data  
let incomes = [|63.0; 16.0; 28.0; 55.0; 22.0; 20.0|]  
let ages = [|38.0; 23.0; 40.0; 27.0; 18.0; 40.0|]  
let willBuy = [|true; false; true; true; false; false|]  
let dataLen = willBuy.Length  
let xdata = Array.init dataLen (fun i -> Vector.FromArray([|incomes.[i]; ages.[i]; 1.0|]))  

// Binding the data  
x.ObservedValue <- xdata  
y.ObservedValue <- willBuy  
len.ObservedValue <- dataLen  

// Inferring the weights  
let ie = InferenceEngine(ExpectationPropagation())  
let wPosterior = ie.Infer<VectorGaussian>(w)  
printfn "Dist over w =\n%A" wPosterior  

// Prediction  
let incomesTest = [|58.0; 18.0; 22.0|]  
let agesTest = [|36.0; 24.0; 37.0|]  
let testDataLen = incomesTest.Length  
let xtestData = Array.init testDataLen (fun i -> Vector.FromArray([|incomesTest.[i]; agesTest.[i]; 1.0|]))  
let jtest = Range(testDataLen)  
let xtest = Variable.Observed<Vector>(xtestData, jtest)  
let wtest = Variable.Random<Vector>(wPosterior)  

let ytest = Variable.AssignVariableArray (Variable.Array<bool>(jtest)) jtest ( fun j -> Variable.IsPositive(Variable.GaussianFromMeanAndVariance  
                            (Variable.InnerProduct(wtest,xtest.[j]), noise)))let ypred = Inference.InferBernoulliArray (ie, ytest)printfn "Output =\n%A" ypred
```
