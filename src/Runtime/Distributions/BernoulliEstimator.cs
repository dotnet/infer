// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a Bernoulli distribution from samples.
    /// </summary>
    public class BernoulliEstimator : Estimator<Bernoulli>, Accumulator<Bernoulli>, Accumulator<bool>, SettableTo<BernoulliEstimator>, ICloneable
    {
        /// <summary>
        /// Number of samples
        /// </summary>
        public long N;

        /// <summary>
        /// Number of samples with value true
        /// </summary>
        public double NProbTrue;

        /// <summary>
        /// Gets the estimated distribution
        /// </summary>
        /// <param name="result">A place to put the resulting distribution. This is ignored because Bernoulli is a struct</param>
        /// <returns>The estimated distribution</returns>
        public Bernoulli GetDistribution(Bernoulli result)
        {
            if (object.ReferenceEquals(result, null)) result = new Bernoulli();
            if (N == 0) return Bernoulli.Uniform();
            result.SetProbTrue(NProbTrue/N);
            return result;
        }

        /// <summary>
        /// Adds a distribution item to the estimator
        /// </summary>
        /// <param name="distribution">A Bernoulli distribution</param>
        public void Add(Bernoulli distribution)
        {
            NProbTrue += distribution.GetProbTrue();
            N++;
        }

        /// <summary>
        /// Adds a sample to the estimator
        /// </summary>
        /// <param name="sample">The sample - true or false</param>
        public void Add(bool sample)
        {
            NProbTrue += sample ? 1 : 0;
            N++;
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        public void Clear()
        {
            N = 0;
            NProbTrue = 0;
        }

        /// <summary>
        /// Returns a copy of the estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            BernoulliEstimator result = new BernoulliEstimator();
            result.SetTo(this);
            return result;
        }

        /// <summary>
        /// Sets this estimator's state from the supplied estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(BernoulliEstimator value)
        {
            N = value.N;
            NProbTrue = value.NProbTrue;
        }
    }
}