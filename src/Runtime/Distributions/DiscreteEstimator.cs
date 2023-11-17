// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a discrete distribution from samples.
    /// </summary>
    public class DiscreteEstimator : Estimator<Discrete>, Accumulator<Discrete>, Accumulator<int>,
                                     SettableTo<DiscreteEstimator>, ICloneable
    {
        /// <summary>
        /// Number of samples
        /// </summary>
        public double N;

        /// <summary>
        /// Vector of counts for each domain value
        /// </summary>
        public Vector NProb;

        /// <summary>
        /// Dimension of the discrete distribution
        /// </summary>
        public int Dimension
        {
            get { return NProb.Count; }
        }

        /// <summary>
        /// Gets the estimated distribution
        /// </summary>
        /// <param name="result">Where to put the estimated distribution</param>
        /// <returns>The estimated distribution</returns>
        public Discrete GetDistribution(Discrete result)
        {
            if (result == null) result = Discrete.Uniform(NProb.Count);
            if (N > 0)
                result.SetProbs(NProb*(1.0/N));
            else
                result.SetToUniform();
            return result;
        }

        /// <summary>
        /// Gets the maximum over all dimensions of the variance of the log probability.  Decreases as N increases.
        /// </summary>
        /// <returns></returns>
        public double GetMaximumVarianceOfLog()
        {
            return 1.0 / NProb.Reduce(double.PositiveInfinity, (min, x) => x > 0 ? System.Math.Min(min, x) : min);
        }

        /// <summary>
        /// Adds all items in another estimator to this estimator.
        /// </summary>
        /// <param name="that">Another estimator</param>
        public void Add(DiscreteEstimator that)
        {
            NProb.SetToSum(NProb, that.NProb);
            N += that.N;
        }

        /// <summary>
        /// Adds a discrete distribution item to the estimator
        /// </summary>
        /// <param name="distribution">A Discrete instance</param>
        public void Add(Discrete distribution)
        {
            NProb.SetToSum(NProb, distribution.GetProbs());
            N++;
        }

        /// <summary>
        /// Adds a weighted discrete distribution item to the estimator
        /// </summary>
        /// <param name="distribution">A Discrete instance</param>
        /// <param name="weight">The weight of the instance</param>
        public void Add(Discrete distribution, double weight)
        {
            NProb.SetToSum(1.0, NProb, weight, distribution.GetProbs());
            N += weight;
        }

        /// <summary>
        /// Adds a weighted discrete sample to the estimator
        /// </summary>
        /// <param name="sample">The sample value</param>
        /// <param name="weight">The weight of the sample</param>
        public void Add(int sample, double weight)
        {
            NProb[sample] += weight;
            N += weight;
        }

        /// <summary>
        /// Adds an discrete sample to the estimator
        /// </summary>
        /// <param name="sample">The sample value</param>
        public void Add(int sample)
        {
            NProb[sample]++;
            N++;
        }

        /// <summary>
        /// Creates a new discrete distribution estimator
        /// </summary>
        /// <param name="dimension">Dimension</param>
        public DiscreteEstimator(int dimension)
        {
            NProb = Vector.Zero(dimension);
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        public void Clear()
        {
            N = 0;
            NProb.SetAllElementsTo(0.0);
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(DiscreteEstimator value)
        {
            N = value.N;
            NProb.SetTo(value.NProb);
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            DiscreteEstimator result = new DiscreteEstimator(Dimension);
            result.SetTo(this);
            return result;
        }
    }
}