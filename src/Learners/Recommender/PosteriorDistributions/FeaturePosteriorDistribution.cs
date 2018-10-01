// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System.Collections.Generic;

    using Distributions;

    /// <summary>
    /// Represents the posterior distribution over feature weights.
    /// </summary>
    public class FeaturePosteriorDistribution
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FeaturePosteriorDistribution"/> class.
        /// </summary>
        /// <param name="traitWeights">The trait weights.</param>
        /// <param name="biasWeight">The bias weight.</param>
        public FeaturePosteriorDistribution(IList<Gaussian> traitWeights, Gaussian biasWeight)
        {
            this.TraitWeights = traitWeights;
            this.BiasWeight = biasWeight;
        }

        /// <summary>
        /// Gets the trait weights.
        /// </summary>
        public IList<Gaussian> TraitWeights { get; }

        /// <summary>
        /// Gets the bias weight.
        /// </summary>
        public Gaussian BiasWeight { get; }
    }
}
