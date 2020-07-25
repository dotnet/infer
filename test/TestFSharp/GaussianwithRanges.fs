// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#light

namespace TestFSharp

open System
open Microsoft.ML.Probabilistic
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Math
open Microsoft.ML.Probabilistic.FSharp

//----------------------------------------------------------------------------------
// Infer.NET: F# script for learning a Gaussian from data
//----------------------------------------------------------------------------------

// The model
module ranges = 
    let learningAGaussianWithRanges() = 
        Console.WriteLine("\n======================Running Gaussian Ranges Tutorial=======================\n");
        // The model
        let len = Variable.New<int>()
        let dataRange = Range(len)
        let mean = Variable.GaussianFromMeanAndVariance(0.0, 100.0)
        let precision = Variable.GammaFromShapeAndScale(1.0, 1.0)
        let x = Variable.AssignVariableArray 
                    (Variable.Array<float>(dataRange))  
                     dataRange (fun d -> Variable.GaussianFromMeanAndPrecision(mean, precision))

        // The data
        let data = Array.init 100 (fun _ -> Rand.Normal(0.0, 1.0))

        // Binding the data
        len.ObservedValue <- data.Length
        x.ObservedValue <- data

        // The inference
        let ie = InferenceEngine(Algorithms.VariationalMessagePassing())
        printfn "mean = %A" (ie.Infer<Gaussian>(mean))
        printfn "prec = %A" (ie.Infer<Gamma>(precision))
