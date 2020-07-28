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
// Infer.NET: F# script for two coins tutorial
//-----------------------------------------------------------------------------------

module coins = 
    let twoCoinsTutorial() = 
        Console.WriteLine("\n========================Running Two Coins Tutorial========================\n");
        let firstCoin = Variable.Bernoulli(0.5)
        let secondCoin = Variable.Bernoulli(0.5)
        let bothHeads = firstCoin &&& secondCoin

        // The inference
        let engine = InferenceEngine()

        let bothHeadsPost = engine.Infer<Bernoulli>(bothHeads)
        printfn "Both heads posterior = %A" bothHeadsPost
        bothHeads.ObservedValue <- false

        let firstCoinPost = engine.Infer<Bernoulli>(firstCoin)
        printfn "First coin posterior = %A" firstCoinPost



