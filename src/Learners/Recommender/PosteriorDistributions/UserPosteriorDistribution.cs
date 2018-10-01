// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners
{
    using System.Collections.Generic;

    using Distributions;

    /// <summary>
    /// Contains the learned parameters for a user.
    /// </summary>
    /// <remarks>
    /// This class is only used for external parameter distribution representation.
    /// </remarks>
    public class UserPosteriorDistribution : EntityPosteriorDistribution
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="UserPosteriorDistribution"/> class.
        /// </summary>
        /// <param name="traits">The user traits.</param>
        /// <param name="bias">The user bias.</param>
        /// <param name="thresholds">The user thresholds.</param>
        public UserPosteriorDistribution(IList<Gaussian> traits, Gaussian bias, IList<Gaussian> thresholds)
            : base(traits, bias)
        {
            this.Thresholds = thresholds;
        }

        /// <summary>
        /// Gets the user thresholds.
        /// </summary>
        public IList<Gaussian> Thresholds { get; }
    }
}
