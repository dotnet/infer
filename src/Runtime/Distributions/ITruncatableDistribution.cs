// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    public interface ITruncatableDistribution<T> : CanGetProbLessThan<T>, CanGetQuantile<T>
    {
        /// <summary>
        /// Returns the distribution of values restricted to an interval.
        /// </summary>
        /// <param name="lowerBound">Inclusive</param>
        /// <param name="upperBound">Exclusive</param>
        /// <returns></returns>
        ITruncatableDistribution<T> Truncate(T lowerBound, T upperBound);
    }
}
