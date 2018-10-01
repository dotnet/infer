---
layout: default 
--- 
[Infer.NET user guide](index.md) : [FSharp Wrapper](FSharp Wrapper.md)

 

## The truncated Gaussian example in F\#

For a description of this tutorial and the C# code please see [the truncated Gaussian tutorial](Truncated Gaussian tutorial.md).

#### F# script

```fsharp
#light

// Reference the Infer.NET DLLs  
#r @"C:/Program Files/Microsoft Research/Infer.NET 2.4/bin/Release/Microsoft.ML.Probabilistic.Compiler.dll"  
#r @"C:/Program Files/Microsoft Research/Infer.NET 2.4/bin/Release/Microsoft.ML.Probabilistic.dll"  
#r @"C:/Program Files/Microsoft Research/Infer.NET 2.4/bin/Release/Microsoft.ML.Probabilistic.FSharp.dll"

open Microsoft.ML.Probabilistic  
open Microsoft.ML.Probabilistic.Models  
open Microsoft.ML.Probabilistic.Distributions  
open Microsoft.ML.Probabilistic.Factors  
open Microsoft.ML.Probabilistic.Math  
open Microsoft.ML.Probabilistic.FSharp

//-----------------------------------------------------------------------------------  
// Infer.NET: F# script for a truncated Gaussian with a variable threshold  
//-----------------------------------------------------------------------------------

// The Model  
let threshold = (Variable.New<float>()).Named("threshold")  
let x = Variable.GaussianFromMeanAndVariance(0.0,1.0).Named("x")  

do Variable.ConstrainTrue(x >> threshold)

// The inference, looping over different thresholds  
let ie = InferenceEngine()  
ie.ShowProgress <- false  
threshold.ObservedValue <- -0.1

for i = 0 to 10 do   
  threshold.ObservedValue <- threshold.ObservedValue + 0.1)  
  printfn "Dist over x given thresh of %A = %A" threshold.ObservedValue (ie.Infer<Gaussian>(x))
```
