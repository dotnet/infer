// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System.Collections.Generic;

    using Distributions;

    /// <summary>
    /// Contains the learned parameters for an item.
    /// </summary>
    /// <remarks>
    /// This class is only used for external parameter distribution representation.
    /// </remarks>
    public class ItemPosteriorDistribution : EntityPosteriorDistribution
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ItemPosteriorDistribution"/> class.
        /// </summary>
        /// <param name="traits">The item traits.</param>
        /// <param name="bias">The item bias.</param>
        public ItemPosteriorDistribution(IList<Gaussian> traits, Gaussian bias)
            : base(traits, bias)
        {
        }
    }
}
