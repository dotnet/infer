// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a Beta distribution from samples.
    /// </summary>
    public class BetaEstimator : Estimator<Beta>, Accumulator<Beta>, Accumulator<double>,
                                 SettableTo<BetaEstimator>, ICloneable
    {
        /// <summary>
        /// Where to accumulate means and variances
        /// </summary>
        public MeanVarianceAccumulator mva;

        /// <summary>
        /// Creates a new Beta estimator
        /// </summary>
        public BetaEstimator()
        {
            mva = new MeanVarianceAccumulator();
        }

        /// <summary>
        /// Gets the estimated distribution
        /// </summary>
        /// <param name="result">Where to put the estimated distribution</param>
        /// <returns>The estimated distribution</returns>
        public Beta GetDistribution(Beta result)
        {
            if (mva.Count == 0) return Beta.Uniform();
            result.SetMeanAndVariance(mva.Mean, mva.Variance);
            return result;
        }

        /// <summary>
        /// Adds a Beta item to the estimator
        /// </summary>
        /// <param name="distribution">A Beta instance</param>
        public void Add(Beta distribution)
        {
            Add(distribution, 1.0);
        }

        /// <summary>
        /// Adds a Beta item to the estimator
        /// </summary>
        /// <param name="distribution">A Beta instance</param>
        /// <param name="weight"></param>
        public void Add(Beta distribution, double weight)
        {
            double x, noiseVariance;
            distribution.GetMeanAndVariance(out x, out noiseVariance);
            mva.Add(x, noiseVariance, weight);
        }        

        /// <summary>
        /// Adds a sample to the estimator
        /// </summary>
        /// <param name="value"></param>
        public void Add(double value)
        {
            mva.Add(value);
        }

        /// <summary>
        /// Adds a sample to the estimator
        /// </summary>
        /// <param name="value"></param>
        /// <param name="weight"></param>
        public void Add(double value, double weight)
        {
            mva.Add(value, weight);
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        public void Clear()
        {
            mva.Clear();
        }

        /// <summary>
        /// Sets the state of this estimator from the supplied estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(BetaEstimator value)
        {
            mva.SetTo(value.mva);
        }


        /// <summary>
        /// Returns a copy of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            BetaEstimator result = new BetaEstimator();
            result.SetTo(this);
            return result;
        }
    }
}