---
layout: default 
--- 
[Infer.NET user guide](index.md) : [FSharp Wrapper](FSharp Wrapper.md)

## Clinical trial example in F\#

For a description of this tutorial and the C# code please see [the clinical trial tutorial](Clinical trial tutorial.md).

#### F# script

```fsharp
#light

// Reference the Infer.NET DLLs  
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
// Infer.NET: F# script for clinical trial example  
//-----------------------------------------------------------------------------------

// Data from a clinical trial  
let controlGroup = Variable.Observed<bool>([|false; false; true; false; false|])  
let treatedGroup = Variable.Observed<bool>([|true; false; true; true; true |])  
let i = controlGroup.Range  
let j = treatedGroup.Range

// Prior on being an effective treatment  
let isEffective = Variable.Bernoulli(0.5).Named("isEffective");  
let probIfTreated = ref (Variable.New<float>())  
let probIfControl = ref (Variable.New<float>())

// If Block function  
let f1 (vb1: Variable<bool>) =  
    probIfControl := Variable.Beta(1.0, 1.0).Named("probIfControl")  
    let controlGroup = Variable.AssignVariableArray   
                           controlGroup i (fun i -> Variable.Bernoulli(!probIfControl)) 
    probIfTreated := Variable.Beta(1.0, 1.0).Named("probIfTreated") 
    let treatedGroup = Variable.AssignVariableArray   
                           treatedGroup j (fun j -> Variable.Bernoulli(!probIfTreated))  
    ()

// IfNot Block function  
let f2 (vb2: Variable<bool>) =    
    let probAll = Variable.Beta(1.0, 1.0).Named("probAll")  
    let controlGroup = Variable.AssignVariableArray   
                           controlGroup i (fun i ->Variable.Bernoulli(probAll))  
    let treatedGroup = Variable.AssignVariableArray  
                           treatedGroup j (fun j ->Variable.Bernoulli(probAll))  
    ()

// Call ifBlock  
Variable.IfBlock isEffective f1 f2  
()

// The inference  
let ie = InferenceEngine()  
printfn "Probability treatment has an effect = %A" (ie.Infer<bool>(isEffective))  
let probIfTreatedPost = ie.Infer<Beta>(!probIfTreated)  
let probIfControlPost = ie.Infer<Beta>(!probIfControl)  
printfn "Probability of good outcome if given treatment = %A" (probIfTreatedPost.GetMean())  
printfn "Probability of good outcome if control = %A" (probIfControlPost.GetMean())
```
