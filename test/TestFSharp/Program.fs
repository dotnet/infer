// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#light
open System
open Microsoft.ML.Probabilistic
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.Factors
open Microsoft.ML.Probabilistic.FSharp

open TwoCoinsTutorial
open TruncatedGaussianTutorial
open GaussianRangesTutorial
open ClinicalTrialTutorial
open BayesPointTutorial
open MixtureGaussiansTutorial

//main Smoke Test .............................................

let _ = TwoCoinsTutorial.coins.twoCoinsTestFunc()
let _ = TruncatedGaussianTutorial.truncated.truncatedTestFunc() 
let _ = GaussianRangesTutorial.ranges.rangesTestFunc()
let _ = ClinicalTrialTutorial.clinical.clinicalTestFunc()
let _ = BayesPointTutorial.bayes.bayesTestFunc()
let _ = MixtureGaussiansTutorial.mixture.mixtureTestFunc()


Console.ReadLine() |> ignore
