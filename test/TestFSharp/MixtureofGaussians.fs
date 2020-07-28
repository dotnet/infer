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

//-----------------------------------------------------------------------------------
// Infer.NET: F# script for a mixture of 2 multivariate Gaussians
//-----------------------------------------------------------------------------------

module mixture = 
    //--------------------------
    // Data for mixture example
    //--------------------------
    let GenerateData nData =
        let trueM1,trueP1 = Vector.FromArray[|2.0;3.0|],PositiveDefiniteMatrix(Array2D.create2D [ [3.0;0.2];[0.2;2.0] ])
        let trueM2,trueP2 = Vector.FromArray[|7.0;5.0|],PositiveDefiniteMatrix(Array2D.create2D [ [2.0;0.4];[0.4;4.0] ])

        let trueVG1 = VectorGaussian.FromMeanAndPrecision(trueM1,trueP1)
        let trueVG2 = VectorGaussian.FromMeanAndPrecision(trueM2,trueP2)
        let truePi = 0.6
        let trueB = new Bernoulli(truePi) 
        Rand.Restart(12347) // Restart the infer.NET random number generator 
        Array.init nData (fun j -> if trueB.Sample() then trueVG1.Sample() else trueVG2.Sample())

    //-----------------------------------
    // The model
    //----------------------------------
    let mixtureOfGaussiansTutorial() = 
        Console.WriteLine("\n=================Running Mixture of Gaussians Tutorial=======================\n");
        // Define a range for the number of mixture components
        let k = Range(2)

        // Mixture component means
        let means = Variable.ArrayInit k (fun k ->
                       Variable.VectorGaussianFromMeanAndPrecision(
                         Vector.Zero(2),
                         PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)))
        // Mixture component precisions
        let precs = Variable.ArrayInit k (fun k ->
                        Variable.WishartFromShapeAndScale(
                            100.0, PositiveDefiniteMatrix.IdentityScaledBy(2,0.01)))

        // Mixture weights 
        let weights = Variable.Dirichlet(k,[|1.0; 1.0|])
        let n = new Range(300)

        // Create latent indicator variable for each data point
        let z = Variable.ArrayInit n (fun i -> Variable.Discrete(weights))

        // Initialise messages randomly so as to break symmetry
        let zinit = Array.init n.SizeAsInt (fun _ -> Discrete.PointMass(Rand.Int(k.SizeAsInt), k.SizeAsInt))
        let _ = z.InitialiseTo(Distribution.Array(zinit))


        // The mixture of Gaussians model
        let data = Variable.ArrayInit n (fun i -> 
                           Variable.SwitchExpr (z.[i]) (fun zi -> 
                                Variable.VectorGaussianFromMeanAndPrecision(means.[zi], precs.[zi])))

        // Binding the data
        data.ObservedValue <- GenerateData(n.SizeAsInt)

        // The inference
        let ie = InferenceEngine(Algorithms.VariationalMessagePassing())

        let wPost = ie.Infer<Dirichlet>(weights) 
        printfn "Estimated means for pi = (%A)" (wPost.GetMean())
        printfn "Distribution over pi = %A" wPost

        let meansPost = Inference.InferVectorGaussianArray(ie, means)
        let precsPost = Inference.InferWishartArray(ie,precs)
        printfn "Distribution over vector Gaussian means =\n%A" meansPost
        printfn "Distribution over vector Gaussian precisions =\n%A" precsPost
        ()