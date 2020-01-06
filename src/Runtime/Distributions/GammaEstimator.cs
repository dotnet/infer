// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a Gamma distribution from samples.
    /// </summary>
    /// <remarks><code>
    /// The distribution is estimated via moment matching (not maximum-likelihood).
    /// In the one-dimensional case,
    /// E[x] = (a+1)/b
    /// var(x) = (a+1)/b^2
    /// b = E[x]/var(x)
    /// a = E[x]^2/var(x) - 1
    /// </code></remarks>
    public class GammaEstimator : Estimator<Gamma>, Accumulator<Gamma>, Accumulator<double>,
                                  SettableTo<GammaEstimator>, ICloneable
    {
        /// <summary>
        /// Where to accumulate means and variances
        /// </summary>
        public MeanVarianceAccumulator mva = new MeanVarianceAccumulator();

        /// <summary>
        /// Retrieves the estimated Gamma
        /// </summary>
        /// <param name="result">Where to put the result</param>
        /// <returns>The resulting distribution</returns>
        public Gamma GetDistribution(Gamma result)
        {
            if (mva.Count == 0) return Gamma.Uniform();
            result.SetMeanAndVariance(mva.Mean, mva.Variance);
            return result;
        }

        /// <summary>
        /// Adds a Gamma distribution item to the estimator
        /// </summary>
        /// <param name="distribution">The distribution instance to add</param>
        public void Add(Gamma distribution)
        {
          Add(distribution, 1.0);
        }

        /// <summary>
        /// Adds a Gamma distribution item to the estimator
        /// </summary>
        /// <param name="distribution">The distribution instance to add</param>
        /// <param name="weight">The weight of the sample</param>
        public void Add(Gamma distribution, double weight)
        {
            double x, noiseVariance;
            distribution.GetMeanAndVariance(out x, out noiseVariance);
            mva.Add(x, noiseVariance, weight);
        }

        /// <summary>
        /// Adds a sample to the estimator
        /// </summary>
        /// <param name="value">The sample to add</param>
        public void Add(double value)
        {
            mva.Add(value);
        }

        /// <summary>
        /// Adds a sample with a given weight to the estimator
        /// </summary>
        /// <param name="value">The sample to add</param>
        /// <param name="weight">The weight of the sample</param>
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
        public void SetTo(GammaEstimator value)
        {
            mva.SetTo(value.mva);
        }

        /// <summary>
        /// Returns a clone of this Gamma estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            GammaEstimator result = new GammaEstimator();
            result.SetTo(this);
            return result;
        }
    }
}