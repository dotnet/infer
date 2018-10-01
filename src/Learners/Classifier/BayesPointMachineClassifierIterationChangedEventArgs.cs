// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System;
    using System.Collections.Generic;

    using Microsoft.ML.Probabilistic.Distributions;

    /// <summary>
    /// Provides information about the training progress of the Bayes point machine classifiers.
    /// </summary>
    public class BayesPointMachineClassifierIterationChangedEventArgs : EventArgs
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BayesPointMachineClassifierIterationChangedEventArgs"/> class 
        /// given the number of iterations of the training algorithm and the posterior distributions of the weights.
        /// </summary>
        /// <param name="completedIterationCount">The number of iterations the training algorithm completed.</param>
        /// <param name="weightPosteriorDistributions">The posterior distributions of the weights.</param>
        public BayesPointMachineClassifierIterationChangedEventArgs(int completedIterationCount, IReadOnlyList<IReadOnlyList<Gaussian>> weightPosteriorDistributions)
        {
            this.CompletedIterationCount = completedIterationCount;
            this.WeightPosteriorDistributions = weightPosteriorDistributions;
        }

        /// <summary>
        /// Gets the number of iterations the training algorithm completed.
        /// </summary>
        public int CompletedIterationCount { get; private set; }

        /// <summary>
        /// Gets the posterior distributions of the weights.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<Gaussian>> WeightPosteriorDistributions { get; private set; }
    }
}
