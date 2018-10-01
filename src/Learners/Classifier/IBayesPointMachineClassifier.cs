// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Distributions;

    /// <summary>
    /// Interface to a Bayes point machine classifier.
    /// </summary>
    /// <typeparam name="TInstanceSource">The type of a source of instances.</typeparam>
    /// <typeparam name="TInstance">The type of an instance.</typeparam>
    /// <typeparam name="TLabelSource">The type of a source of labels.</typeparam>
    /// <typeparam name="TLabel">The type of a label.</typeparam>
    /// <typeparam name="TLabelDistribution">The type of a distribution over labels in standard data format.</typeparam>
    /// <typeparam name="TTrainingSettings">The type of the settings for training.</typeparam>
    /// <typeparam name="TPredictionSettings">The type of the settings for prediction.</typeparam>
    public interface IBayesPointMachineClassifier<in TInstanceSource, in TInstance, in TLabelSource, TLabel, out TLabelDistribution, out TTrainingSettings, out TPredictionSettings> 
        : IPredictor<TInstanceSource, TInstance, TLabelSource, TLabel, TLabelDistribution>, IPredictorIncrementalTraining<TInstanceSource, TLabelSource>, ICustomSerializable
        where TTrainingSettings : BayesPointMachineClassifierTrainingSettings
        where TPredictionSettings : IBayesPointMachineClassifierPredictionSettings<TLabel>
    {
        /// <summary>
        /// The event that is fired at the end of each iteration of the Bayes point machine classifier training algorithm.
        /// </summary>
        event EventHandler<BayesPointMachineClassifierIterationChangedEventArgs> IterationChanged;

        /// <summary>
        /// Gets the settings of the Bayes point machine classifier.
        /// </summary>
        new IBayesPointMachineClassifierSettings<TLabel, TTrainingSettings, TPredictionSettings> Settings { get; }

        /// <summary>
        /// Gets the natural logarithm of the model evidence. Use this for model selection.
        /// </summary>
        double LogModelEvidence { get; }

        /// <summary>
        /// Gets the posterior distributions of the weights of the Bayes point machine classifier.
        /// </summary>
        IReadOnlyList<IReadOnlyList<Gaussian>> WeightPosteriorDistributions { get; }
    }
}
