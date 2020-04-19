// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a point mass distribution from a sample or point mass distribution.
    /// </summary>
    /// <typeparam name="T">Sample type</typeparam>
    public class PointMassEstimator<T> : Estimator<PointMass<T>>, Accumulator<T>, Accumulator<PointMass<T>>
    {
        /// <summary>
        /// The last value added
        /// </summary>
        T point;

        /// <inheritdoc cref="Accumulator{T}.Add"/>
        public void Add(T item)
        {
            point = item;
        }

        /// <inheritdoc cref="Accumulator{T}.Add"/>
        public void Add(PointMass<T> item)
        {
            point = item.Point;
        }

        /// <inheritdoc cref="Accumulator{T}.Clear"/>
        public void Clear()
        {
            point = default;
        }

        /// <inheritdoc cref="Estimator{T}.GetDistribution"/>
        public PointMass<T> GetDistribution(PointMass<T> result)
        {
            return new PointMass<T>(point);
        }
    }
}
