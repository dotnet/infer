// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a Wishart distribution from samples.
    /// </summary>
    /// <remarks><code>
    /// The distribution is estimated via moment matching (not maximum-likelihood).
    /// E[X] = (a+(d+1)/2)/B
    /// var(X_ii) = (a+(d+1)/2)*diag(inv(B))^2
    /// because X_ii ~ Gamma(a+(d+1)/2, 1/diag(inv(B))).
    /// Therefore: 
    /// a = E[X_ii]^2/var(X_ii) - (d+1)/2
    /// B = (a+(d+1)/2)/E[X]
    /// </code></remarks>
    /// In the one-dimensional case,
    /// E[log(x)] = -log(b) + digamma(a+1) 
    ///           =approx -log(b) + log(a+1) - 1/2/(a+1) 
    ///           = log(E[x]) - 1/2/(a+1)
    /// In the Wishart case,
    /// E[logdet(X)] = -logdet(B) + sum_{i=0..d-1} digamma(a + (d+1-i)/2)
    ///              =approx -logdet(B) + d*digamma(a+(d+1)/2)
    ///              =approx -logdet(B) + d*log(a+(d+1)/2) - d/2/(a+(d+1)/2) 
    ///              = log(E[X}) - d/2/(a+(d+1)/2)
    public class WishartEstimator : Estimator<Wishart>, Accumulator<Wishart>, Accumulator<PositiveDefiniteMatrix>,
                                    SettableTo<WishartEstimator>, ICloneable
    {
        /// <summary>
        /// Where to accumulate mean and variance matrices
        /// </summary>
        public MatrixMeanVarianceAccumulator mva;

        /// <summary>
        /// Creates a new Wishart estimator
        /// </summary>
        /// <param name="dimension">The dimension of the Wishart distribution</param>
        public WishartEstimator(int dimension)
        {
            mva = new MatrixMeanVarianceAccumulator(dimension, dimension);
        }

        /// <summary>
        /// The dimension of the Wishart distribution
        /// </summary>
        public int Dimension
        {
            get { return mva.Mean.Rows; }
        }

        /// <summary>
        /// Retrieves the Wishart estimator
        /// </summary>
        /// <param name="result">Where to put the result</param>
        /// <returns>The resulting distribution</returns>
        public Wishart GetDistribution(Wishart result)
        {
            if (result == null) result = new Wishart(Dimension);
            if (mva.Count == 0) result.SetToUniform();
            else result.SetMeanAndVariance(new PositiveDefiniteMatrix(mva.Mean), mva.Variance);
            return result;
        }

        /// <summary>
        /// Adds a Wishart distribution item to the estimator
        /// </summary>
        /// <param name="item">The distribution instance to add</param>
        public void Add(Wishart item)
        {
          Add(item, 1.0);
        }
        public void Add(Wishart item, double weight)
        {
            PositiveDefiniteMatrix x = new PositiveDefiniteMatrix(Dimension, Dimension);
            PositiveDefiniteMatrix noiseVariance = new PositiveDefiniteMatrix(Dimension, Dimension);
            item.GetMeanAndVariance(x, noiseVariance);
            mva.Add(x, noiseVariance, weight);
        }

        /// <summary>
        /// Adds a sample to the estimator
        /// </summary>
        /// <param name="item">The sample to add</param>
        public void Add(PositiveDefiniteMatrix item)
        {
          Add(item, 1.0);
        }
        public void Add(PositiveDefiniteMatrix item, double weight)
        {
            mva.Add(item, weight);
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        public void Clear()
        {
            mva.Clear();
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(WishartEstimator value)
        {
            mva.SetTo(value.mva);
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            WishartEstimator result = new WishartEstimator(Dimension);
            result.SetTo(this);
            return result;
        }
    }

#if false
    /// <summary>
    /// Estimate a Wishart distribution from samples.
    /// </summary>
    /// <remarks><code>
    /// The distribution is estimated via moment matching (not maximum-likelihood).
    /// In the one-dimensional case,
    /// E[log(x)] = -log(b) + digamma(a+1) 
    ///           =approx -log(b) + log(a+1) - 1/2/(a+1) 
    ///           = log(E[x]) - 1/2/(a+1)
    /// In the Wishart case,
    /// E[logdet(X)] = -logdet(B) + sum_{i=0..d-1} digamma(a + (d+1-i)/2)
    ///              =approx -logdet(B) + d*digamma(a+(d+1)/2)
    ///              =approx -logdet(B) + d*log(a+(d+1)/2) - d/2/(a+(d+1)/2) 
    ///              = log(E[X}) - d/2/(a+(d+1)/2)
    /// </code></remarks>
    public class WishartEstimator : Estimator<Wishart>, Accumulator<Wishart>, Accumulator<PositiveDefiniteMatrix>
    {
        public double N;
        public PositiveDefiniteMatrix Sum;
        public double SumLogDet;

        public int Dimension { get { return Sum.Rows; } }

        public Wishart GetDistribution(Wishart result)
        {
            if (result == null) result = new Wishart(Dimension);
            PositiveDefiniteMatrix m = new PositiveDefiniteMatrix(Sum);
            m.Scale(1.0 / N);
            double lm = m.LogDeterminant() - SumLogDet / N;
            if (lm < 1e-10) return Wishart.PointMass(m);
            double ad = 0.5 * Dimension / lm;
            result.Precision = ad - 0.5 * (Dimension + 1.0);
            m.Scale(1.0 / ad);
            result.PrecisionTimesMean.SetToInverse(m);
            return result;
        }

        public void Add(Wishart item)
        {
            PositiveDefiniteMatrix m = item.GetMean();
            Sum.SetToSum(Sum, m);
            SumLogDet += item.MeanLogDeterminant();
            N++;
        }

        public void Add(PositiveDefiniteMatrix item)
        {
            Sum.SetToSum(Sum, item);
            SumLogDet += item.LogDeterminant();
            N++;
        }

        public WishartEstimator(int dimension)
        {
            Sum = new PositiveDefiniteMatrix(dimension, dimension);
        }
    }
#endif
}