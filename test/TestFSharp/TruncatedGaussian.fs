// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#light

namespace TestFSharp

open System
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.FSharp

//-----------------------------------------------------------------------------------
// Infer.NET: F# script for a truncated Gaussian with a variable threshold
//-----------------------------------------------------------------------------------

// The Model
module truncated =
    let truncatedGaussianTutorial() = 
        Console.WriteLine("\n====================Running Truncated Gaussian Tutorial======================\n");
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
        
