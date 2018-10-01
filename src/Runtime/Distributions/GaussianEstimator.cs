// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a Gaussian distribution from samples.
    /// </summary>
    public class GaussianEstimator : Estimator<Gaussian>, Accumulator<Gaussian>, Accumulator<double>,
                                     SettableTo<GaussianEstimator>, ICloneable
    {
        /// <summary>
        /// Where to accumulate means and variances
        /// </summary>
        public MeanVarianceAccumulator mva;

        /// <summary>
        /// Creates a new Gaussian estimator
        /// </summary>
        public GaussianEstimator()
        {
            mva = new MeanVarianceAccumulator();
        }

        /// <summary>
        /// Adds a Gaussian distribution item to the estimator
        /// </summary>
        /// <param name="distribution">The distribution instance to add</param>
        public void Add(Gaussian distribution)
        {
            Add(distribution, 1.0);
        }

        /// <summary>
        /// Adds a Gaussian distribution with given weight to the estimator
        /// </summary>
        /// <param name="distribution">The distribution instance to add</param>
        /// <param name="weight">The weight of the distribution</param>
        public void Add(Gaussian distribution, double weight)
        {
            double x, noiseVariance;
            distribution.GetMeanAndVariance(out x, out noiseVariance);
            mva.Add(x, noiseVariance, weight);
        }

        /// <summary>
        /// Adds an sample to the estimator
        /// </summary>
        /// <param name="sample">The sample add</param>
        public void Add(double sample)
        {
            mva.Add(sample);
        }

        /// <summary>
        /// Adds a sample with a given weight to the estimator
        /// </summary>
        /// <param name="sample">The sample to add</param>
        /// <param name="weight">The weight of the sample</param>
        public void Add(double sample, double weight)
        {
            mva.Add(sample, weight);
        }

        /// <summary>
        /// Computes the maximum-likelihood Gaussian from the samples.
        /// </summary>
        /// <param name="result"></param>
        /// <returns>Returns a new Gaussian object.</returns>
        public Gaussian GetDistribution(Gaussian result)
        {
            if (mva.Count == 0) return Gaussian.Uniform();
            result.SetMeanAndVariance(mva.Mean, mva.Variance);
            return result;
        }

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        public void Clear()
        {
            mva.Clear();
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(GaussianEstimator value)
        {
            mva.SetTo(value.mva);
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            GaussianEstimator result = new GaussianEstimator();
            result.SetTo(this);
            return result;
        }
    }
}