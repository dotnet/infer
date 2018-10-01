---
layout: default 
--- 
[Infer.NET user guide](index.md) : [FSharp Wrapper](FSharp Wrapper.md) 

## Creating delegates in F\#

Some functions, such as Variable.Factor, require passing a delegate. Unfortunately these functions are not compatible with the default method of creating a delegate in F#. Instead you need to create the delegate using our provided _createDelegate_ function. For example, consider the following C# code from the user guide:

```csharp
Variable<double> x = Variable<double>.Factor(Factor.Gaussian, randomMean, randomPrecision);
```

This should be converted into F# as follows:

```fsharp
let d = (Microsoft.ML.Probabilistic.FSharp.Factors.createDelegate <@ Factor.Gaussian(0.0,0.0) @>) :?> FactorMethod<double,double,double>

let x = Variable<double>.Factor(d, randomMean, randomPrecision);
```

Note that the argument to createDelegate is a quotation of a fully specified function call, with arguments of known type to resolve ambiguities.
