// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#light
open System

//main Smoke Test .............................................

let _ = TwoCoinsTutorial.coins.twoCoinsTestFunc()
let _ = TruncatedGaussianTutorial.truncated.truncatedTestFunc() 
let _ = GaussianRangesTutorial.ranges.rangesTestFunc()
let _ = ClinicalTrialTutorial.clinical.clinicalTestFunc()
let _ = BayesPointTutorial.bayes.bayesTestFunc()
let _ = MixtureGaussiansTutorial.mixture.mixtureTestFunc()
let _ = DifficultyAbilityExample.DifficultyAbility.main()

Console.ReadLine() |> ignore
