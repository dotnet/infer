// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a TruncatedGaussian distribution from samples.
    /// </summary>
    public class TruncatedGaussianEstimator : Estimator<TruncatedGaussian>, Accumulator<TruncatedGaussian>,
                                              SettableTo<TruncatedGaussianEstimator>, ICloneable
    {
        /// <summary>
        /// Where to accumulate means and variances
        /// </summary>
        public MeanVarianceAccumulator mva;

        public double minLowerBound, maxUpperBound;

        /// <summary>
        /// Creates a new TruncatedGaussian estimator
        /// </summary>
        public TruncatedGaussianEstimator()
        {
            mva = new MeanVarianceAccumulator();
            minLowerBound = double.PositiveInfinity;
            maxUpperBound = double.NegativeInfinity;
        }

        /// <summary>
        /// Adds a TruncatedGaussian distribution item to the estimator
        /// </summary>
        /// <param name="distribution">The distribution instance to add</param>
        public void Add(TruncatedGaussian distribution)
        {
            minLowerBound = System.Math.Min(minLowerBound, distribution.LowerBound);
            maxUpperBound = System.Math.Max(maxUpperBound, distribution.UpperBound);
            double x, noiseVariance;
            distribution.GetMeanAndVariance(out x, out noiseVariance);
            mva.Add(x, noiseVariance, 1.0);
        }

        /// <summary>
        /// Computes the maximum-likelihood TruncatedGaussian from the samples.
        /// </summary>
        /// <param name="result"></param>
        /// <returns>Returns a new TruncatedGaussian object.</returns>
        public TruncatedGaussian GetDistribution(TruncatedGaussian result)
        {
            if (mva.Count == 0) return TruncatedGaussian.Uniform();
            return new TruncatedGaussian(new Gaussian(mva.Mean, mva.Variance));
        }

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        public void Clear()
        {
            mva.Clear();
            minLowerBound = double.PositiveInfinity;
            maxUpperBound = double.NegativeInfinity;
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(TruncatedGaussianEstimator value)
        {
            mva.SetTo(value.mva);
            minLowerBound = value.minLowerBound;
            maxUpperBound = value.maxUpperBound;
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            TruncatedGaussianEstimator result = new TruncatedGaussianEstimator();
            result.SetTo(this);
            return result;
        }
    }
}