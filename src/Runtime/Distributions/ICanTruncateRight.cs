// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;

    /// <summary>
    /// Whether the distribution can be right truncated.
    /// </summary>
    public interface ICanTruncateRight<TDomain>
        where TDomain : IComparable<TDomain>
    {
        /// <summary>
        /// Gets the end point of the truncated distribution.
        /// </summary>
        /// <returns>The end point.</returns>
        TDomain GetEndPoint();

        /// <summary>
        /// Truncates the distribution at the given point.
        /// </summary>
        /// <param name="endPoint">All domain values greater than this are guaranteed to have zero probability.</param>
        void TruncateRight(TDomain endPoint);
    }
}
