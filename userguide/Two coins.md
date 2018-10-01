---
layout: default 
--- 
[Infer.NET user guide](index.md) : [FSharp Wrapper](FSharp Wrapper.md)

## The two coins example in F\#

For a description of this tutorial and the C# code please see [the two coins tutorial](Two coins tutorial.md).

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
open Microsoft.ML.Probabilistic.FSharp

//-----------------------------------------------------------------------------------  
// Infer.NET: F# script for the two coins example  
//-----------------------------------------------------------------------------------

// The model  
let firstCoin = Variable.Bernoulli(0.5)  
let secondCoin = Variable.Bernoulli(0.5)  
let bothHeads = firstCoin &&& secondCoin

// The inference  
let ie = InferenceEngine()  

let bothHeadsPost = ie.Infer<Bernoulli>(bothHeads)  
printfn "Both heads posterior = %A" bothHeadsPost  
bothHeads.ObservedValue <- false  

let firstCoinPost = ie.Infer<Bernoulli>(firstCoin)  
printfn "First coin posterior = %A" firstCoinPost
```
