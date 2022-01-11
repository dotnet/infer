// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a GammaPower distribution from samples.
    /// </summary>
    /// <remarks><code>
    /// The distribution is estimated via moment matching (not maximum-likelihood).
    /// In the one-dimensional case,
    /// E[x] = (a+1)/b
    /// var(x) = (a+1)/b^2
    /// b = E[x]/var(x)
    /// a = E[x]^2/var(x) - 1
    /// </code></remarks>
    public class GammaPowerEstimator : Estimator<GammaPower>, Accumulator<GammaPower>, Accumulator<double>,
                                       SettableTo<GammaPowerEstimator>, ICloneable
    {
        /// <summary>
        /// Inner estimator
        /// </summary>
        private GammaEstimator gammaEstimator = new GammaEstimator();

        /// <summary>
        /// Desired power
        /// </summary>
        public readonly double Power;

        /// <summary>
        /// Creates a new GammaPower estimator
        /// </summary>
        public GammaPowerEstimator(double power)
        {
            this.Power = power;
        }

        /// <summary>
        /// Retrieves the estimated GammaPower
        /// </summary>
        /// <param name="result">Where to put the result</param>
        /// <returns>The resulting distribution</returns>
        public GammaPower GetDistribution(GammaPower result)
        {
            return GammaPower.FromGamma(gammaEstimator.GetDistribution(new Gamma()), Power);
        }

        /// <summary>
        /// Adds a GammaPower distribution item to the estimator
        /// </summary>
        /// <param name="distribution">The distribution instance to add</param>
        public void Add(GammaPower distribution)
        {
            Add(distribution, 1.0);
        }

        /// <summary>
        /// Adds a GammaPower distribution item to the estimator
        /// </summary>
        /// <param name="distribution">The distribution instance to add</param>
        /// <param name="weight">The weight of the sample</param>
        public void Add(GammaPower distribution, double weight)
        {
            gammaEstimator.Add(Gamma.FromShapeAndRate(distribution.Shape, distribution.Rate), weight);
        }

        /// <summary>
        /// Adds a sample to the estimator
        /// </summary>
        /// <param name="value">The sample to add</param>
        public void Add(double value)
        {
            gammaEstimator.Add(System.Math.Pow(value, 1/Power));
        }

        /// <summary>
        /// Adds a sample with a given weight to the estimator
        /// </summary>
        /// <param name="value">The sample to add</param>
        /// <param name="weight">The weight of the sample</param>
        public void Add(double value, double weight)
        {
            gammaEstimator.Add(System.Math.Pow(value, 1/Power), weight);
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        public void Clear()
        {
            gammaEstimator.Clear();
        }

        /// <summary>
        /// Sets the state of this estimator from the supplied estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(GammaPowerEstimator value)
        {
            if (this.Power != value.Power) throw new ArgumentException($"Incompatible powers: this.Power={Power}, value.Power={value.Power}", nameof(value));
            gammaEstimator.SetTo(value.gammaEstimator);
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            GammaPowerEstimator result = new GammaPowerEstimator(Power);
            result.SetTo(this);
            return result;
        }
    }
}