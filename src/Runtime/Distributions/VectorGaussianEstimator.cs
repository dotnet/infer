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
    public class VectorGaussianEstimator : Estimator<VectorGaussian>, Accumulator<VectorGaussian>, Accumulator<Vector>,
                                           SettableTo<VectorGaussianEstimator>, ICloneable
    {
        private Vector x;
        private PositiveDefiniteMatrix noiseVariance;
        private VectorMeanVarianceAccumulator mva;

        /// <summary>
        /// Dimension of the VectorGaussian
        /// </summary>
        public int Dimension
        {
            get { return x.Count; }
        }

        /// <summary>
        /// Creates a new VectorGaussian estimator of a given dimension
        /// </summary>
        /// <param name="dimension">The dimension</param>
        public VectorGaussianEstimator(int dimension)
        {
            x = Vector.Zero(dimension);
            noiseVariance = new PositiveDefiniteMatrix(dimension, dimension);
            mva = new VectorMeanVarianceAccumulator(dimension);
        }

        /// <summary>
        /// Adds a VectorGaussian distribution item to the estimator
        /// </summary>
        /// <param name="distribution">The distribution instance to add</param>
        public void Add(VectorGaussian distribution)
        {
            Add(distribution, 1.0);
        }

        /// <summary>
        /// Adds a weighted VectorGaussian distribution item to the estimator
        /// </summary>
        /// <param name="distribution">The distribution instance to add</param>
        /// <param name="weight">The weight of the distribution</param>
        public void Add(VectorGaussian distribution, double weight)
        {
            distribution.GetMeanAndVariance(x, noiseVariance);
            // noiseVariance will be modified
            mva.Add(x, noiseVariance, weight);
        }

        /// <summary>
        /// Adds a sample item to the estimator
        /// </summary>
        /// <param name="sample">The sample value to add</param>
        public void Add(Vector sample)
        {
            mva.Add(sample);
        }

        /// <summary>
        /// Add a sample item with a given weight to the estimator
        /// </summary>
        /// <param name="sample">The sample value to add</param>
        /// <param name="weight">The weight of the sample</param>
        public void Add(Vector sample, double weight)
        {
            mva.Add(sample, weight);
        }

        /// <summary>
        /// Computes the maximum-likelihood Gaussian from the samples.
        /// </summary>
        /// <param name="result">May be null.</param>
        /// <returns>If result is not null, modifies and returns result.  
        /// Otherwise returns a new Gaussian object.</returns>
        public VectorGaussian GetDistribution(VectorGaussian result)
        {
            if (result == null) result = new VectorGaussian(Dimension);
            if (mva.Count == 0) result.SetToUniform();
            else result.SetMeanAndVariance(mva.Mean, mva.Variance);
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
        public void SetTo(VectorGaussianEstimator value)
        {
            mva.SetTo(value.mva);
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            VectorGaussianEstimator result = new VectorGaussianEstimator(Dimension);
            result.SetTo(this);
            return result;
        }
    }
}