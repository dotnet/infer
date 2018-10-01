// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.BayesPointMachineClassifierInternal
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Distributions;

    /// <summary>
    /// Interface to both training and prediction algorithms of a Bayes point machine classifier with factorized weight distributions.
    /// </summary>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TLabelDistribution">The type of a distribution over labels.</typeparam>
    internal interface IInferenceAlgorithms<in TLabel, out TLabelDistribution> : ICustomSerializable
    {
        /// <summary>
        /// The event that is fired at the end of each iteration of the training algorithm.
        /// </summary>
        event EventHandler<BayesPointMachineClassifierIterationChangedEventArgs> IterationChanged;

        /// <summary>
        /// Gets a value indicating whether the inference algorithms expect features in a sparse representation.
        /// </summary>
        bool UseSparseFeatures { get; }

        /// <summary>
        /// Gets the total number of features that the inference algorithms use.
        /// </summary>
        int FeatureCount { get; }

        /// <summary>
        /// Gets the distributions over weights as factorized <see cref="Gaussian"/> distributions.
        /// </summary>
        IReadOnlyList<IReadOnlyList<Gaussian>> WeightDistributions { get; }

        /// <summary>
        /// Sets the number of batches the training data is split into and resets the per-batch output messages.
        /// </summary>
        /// <param name="value">The number of batches to use.</param>
        void SetBatchCount(int value);

        /// <summary>
        /// Runs the training algorithm for the specified features and labels.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <param name="labels">The labels.</param>
        /// <param name="iterationCount">The number of iterations to run the training algorithm for.</param>
        /// <param name="batchNumber">
        /// An optional batch number. Defaults to 0 and is used only if the training data is divided into batches.
        /// </param>
        /// <returns>
        /// The natural logarithm of the evidence for the Bayes point machine classifier model 
        /// if the training algorithm computes evidence and 0 otherwise.
        /// </returns>
        double Train(double[][] featureValues, int[][] featureIndexes, TLabel[] labels, int iterationCount, int batchNumber = 0);

        /// <summary>
        /// Runs the prediction algorithm for the specified features.
        /// </summary>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="featureIndexes">The feature indexes.</param>
        /// <param name="iterationCount">The number of iterations to run the prediction algorithm for.</param>
        /// <returns>The predictive distributions over labels.</returns>
        IEnumerable<TLabelDistribution> PredictDistribution(double[][] featureValues, int[][] featureIndexes, int iterationCount);
    }
}