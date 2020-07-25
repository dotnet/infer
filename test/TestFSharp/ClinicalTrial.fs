// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

#light

namespace TestFSharp

open System
open Microsoft.ML.Probabilistic.Models
open Microsoft.ML.Probabilistic.Distributions
open Microsoft.ML.Probabilistic.FSharp
//------------------------------------------------------------------------------------

//-----------------------------------------------------------------------------------
// Infer.NET: F# script for clinical trial example
//-----------------------------------------------------------------------------------

// Data from a clinical trial
module clinical = 
    let controlGroup = Variable.Observed<bool>([|false; false; true; false; false|])
    let treatedGroup = Variable.Observed<bool>([|true; false; true; true; true |])
    let i = controlGroup.Range
    let j = treatedGroup.Range

    // Prior on being an effective treatment
    let isEffective = Variable.Bernoulli(0.5).Named("isEffective");
    let probIfTreated = ref (Variable.New<float>())
    let probIfControl = ref (Variable.New<float>())

    // If block
    let f1 (vb1: Variable<bool>) = 
        probIfControl := Variable.Beta(1.0, 1.0).Named("probIfControl") 
        let controlGroup = Variable.AssignVariableArray 
                               controlGroup i (fun i ->Variable.Bernoulli(!probIfControl))
        probIfTreated := Variable.Beta(1.0, 1.0).Named("probIfTreated")
        let treatedGroup = Variable.AssignVariableArray 
                               treatedGroup j (fun j ->Variable.Bernoulli(!probIfTreated))
        ()
                               
    // IfNot Block function
    let f2 (vb2: Variable<bool>) = 
        let probAll = Variable.Beta(1.0, 1.0).Named("probAll") 
        let controlGroup = Variable.AssignVariableArray
                               controlGroup i (fun i ->Variable.Bernoulli(probAll))
        let treatedGroup = Variable.AssignVariableArray
                               treatedGroup j (fun j ->Variable.Bernoulli(probAll))
        ()
    
    // The model
    Variable.IfBlock isEffective f1 f2
    ()

    let clinicalTrialTutorial() = 
            Console.WriteLine("\n====================Running Clinical Trial Tutorial==========================\n");
            
            // The inference
            let ie = InferenceEngine()
            printfn "Probability treatment has an effect = %A" (ie.Infer<Bernoulli>(isEffective))
            let probIfTreatedPost = ie.Infer<Beta>(!probIfTreated)
            let probIfControlPost = ie.Infer<Beta>(!probIfControl)
            printfn "Probability of good outcome if given treatment = %A" (probIfTreatedPost.GetMean())
            printfn "Probability of good outcome if control = %A" (probIfControlPost.GetMean())

