---
layout: default
---

[Infer.NET user guide](index.md) : [The Infer.NET Modelling API](FSharp Wrapper.md)

## Arrays in F\#

The module `Array2D` provides a method for converting a list of lists into a 2-Dimensional .NET array. `Array2D.Create2D` creates a 2-Dimensional array `M` from two lists representing vectors `V1`,`V2`, where `M[i][j] = v1[i],v2[j]`. An example of its use is shown in [the Mixture of Gaussians tutorial](Mixture of Gaussians.md) below:

```fsharp
//--------------------------
// Data for mixture example
//--------------------------

let GenerateData nData =
    let trueM1,trueP1 = Vector[|2.0;3.0|],PositiveDefiniteMatrix(Array2D.create2D [ [3.0;0.2];[0.2;2.0] ])
    let trueM2,trueP2 = Vector[|7.0;5.0|],PositiveDefiniteMatrix(Array2D.create2D [ [2.0;0.4];[0.4;4.0] ])
    let trueVG1 = VectorGaussian.FromMeanAndPrecision(trueM1,trueP1)
    let trueVG2 = VectorGaussian.FromMeanAndPrecision(trueM2,trueP2)
    let truePi = 0.6
    let trueB = new Bernoulli(truePi) 
    Rand.Restart(12347) // Restart the infer.NET random number generator 
    Array.init nData (fun j -> if trueB.Sample()then trueVG1.Sample() else trueVG2.Sample())
```