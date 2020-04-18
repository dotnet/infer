// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a Poisson distribution from samples.
    /// </summary>
    public class PoissonEstimator : Estimator<Poisson>, Accumulator<Poisson>, Accumulator<double>,
                                    SettableTo<PoissonEstimator>, ICloneable
    {
        #region Estimator<Poisson> Members

        private double Count;
        private double Mean;

        /// <summary>
        /// Retrieves an estimation of the distribution
        /// </summary>
        /// <param name="result"></param>
        /// <returns></returns>
        public Poisson GetDistribution(Poisson result)
        {
            if (Count > 0.0)
                return new Poisson(Mean);
            else
                return Poisson.Uniform();
        }

        #endregion

        #region Accumulator<Poisson> Members

        /// <summary>
        /// Add a distribution sample to the estimator
        /// </summary>
        /// <param name="item">The item to add</param>
        /// <remarks>Assumes a Com-Poisson precision of 1.0</remarks>
        public void Add(Poisson item)
        {
            if (!item.IsPointMass && item.Precision != 1.0)
                throw new InferRuntimeException("Poisson estimator does not support a non-unity precision");
            Add(item.GetMean());
        }

        /// <summary>
        /// Clear the estimator
        /// </summary>
        public void Clear()
        {
            Count = 0.0;
            Mean = 0.0;
        }

        #endregion

        #region Accumulator<double> Members

        /// <summary>
        /// Add a domain sample to the estimator
        /// </summary>
        /// <param name="item"></param>
        public void Add(double item)
        {
            Add(item, 1.0);
        }

        public void Add(double item, double weight)
        {
            Count += weight;
            double diff = item - Mean;
            double s = weight / Count;
            Mean += s * diff;
        }

        #endregion

        #region SettableTo<PoissonEstimator> Members

        /// <summary>
        /// Set this estimator to another
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(PoissonEstimator value)
        {
            Count = value.Count;
            Mean = value.Mean;
        }

        #endregion

        #region ICloneable Members

        /// <summary>
        /// Clone this estimator
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            PoissonEstimator result = new PoissonEstimator();
            result.SetTo(this);
            return result;
        }

        #endregion
    }
}