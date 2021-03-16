// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;

    /// <summary>
    /// Whether the distribution can be left-truncated.
    /// </summary>
    public interface ICanTruncateLeft<TDomain>
    where TDomain : IComparable<TDomain>
    {
        /// <summary>
        /// Gets the start point of the truncated distribution.
        /// </summary>
        /// <returns>The start point.</returns>
        TDomain GetStartPoint();

        /// <summary>
        /// Truncates the distribution at the given point.
        /// </summary>
        /// <param name="startPoint">All values less than this are guaranteed to have zero probability.</param>
        void TruncateLeft(TDomain startPoint);
    }
}
