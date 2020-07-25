// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#light
namespace TestFSharp
open System
open System.Diagnostics
open Xunit

module main =
    [<Fact>]
    let testTutorials() =
        let coreAssemblyInfo = FileVersionInfo.GetVersionInfo(typeof<Object>.Assembly.Location)
        printfn "%s .NET version %s mscorlib %s" (if Environment.Is64BitProcess then "64-bit" else "32-bit") (Environment.Version.ToString ()) coreAssemblyInfo.ProductVersion

        //main Smoke Test .............................................

        coins.twoCoinsTutorial()
        truncated.truncatedGaussianTutorial() 
        ranges.learningAGaussianWithRanges()
        clinical.clinicalTrialTutorial()
        bayes.BayesPointMachineExample()
        mixture.mixtureOfGaussiansTutorial()
        DifficultyAbility.main()

        //Console.ReadLine() |> ignore

    testTutorials()