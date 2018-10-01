// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System.Collections.Generic;

    using Distributions;

    /// <summary>
    /// Contains the learned parameters for an entity (user or item).
    /// </summary>
    /// <remarks>
    /// This class is only used for external parameter distribution representation.
    /// </remarks>
    public abstract class EntityPosteriorDistribution
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EntityPosteriorDistribution"/> class.
        /// </summary>
        /// <param name="traits">The entity traits.</param>
        /// <param name="bias">The entity bias.</param>
        protected EntityPosteriorDistribution(IList<Gaussian> traits, Gaussian bias)
        {
            this.Traits = traits;
            this.Bias = bias;
        }

        /// <summary>
        /// Gets the entity traits.
        /// </summary>
        public IList<Gaussian> Traits { get; }

        /// <summary>
        /// Gets the entity bias.
        /// </summary>
        public Gaussian Bias { get; }
    }
}
