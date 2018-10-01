---
layout: default 
--- 
[Infer.NET user guide](index.md) : [FSharp Wrapper](FSharp Wrapper.md)

## Operator Overloading in F\#

Some Operator Overloads are not recognised by F#, in particular the comparison operators "<.<=,==,>,>=" are not recognised when used to compare `Variable<'a>` with either `Variable<'a>` or type `'a`. For example when trying to compare `Variable<float>` with a float value which occurs in the truncated Gaussian tutorial. The rewritten operators contained in the module Operators and are given names such as " >>" to represent the overloaded Greater Than operator. Other operators are "<<" to represent the overloaded Less Than operator, "==" to represent the overloaded Equality operator, "<<==" to represent the overloaded strictly Less Than operator and ">>==" to represent the overloaded Strictly Greater Than operator. The overloaded Greater than Operator is used in the [Efficient truncated Gaussian tutorial](Efficient truncated Gaussian.md) as shown below:

```fsharp
// The Model  
let threshold = (Variable.New<float>()).Named("threshold")  
let x = Variable.GaussianFromMeanAndVariance(0.0,1.0).Named("x")

do Variable.ConstrainTrue(x >> threshold)

// The inference, looping over different thresholds  
let ie = InferenceEngine()  
ie.ShowProgress <- false  
threshold.ObservedValue <- -0.1  

for i = 0 to 10 do  
threshold.ObservedValue <- threshold.ObservedValue + 0.1   
    printfn "Dist over x given thresh of %A = %A" threshold.ObservedValue (ie.Infer<Gaussian>(x))
```
