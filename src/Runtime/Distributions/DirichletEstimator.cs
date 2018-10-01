// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Estimates a Dirichlet distribution from samples.
    /// </summary>
    public class DirichletEstimator : Estimator<Dirichlet>, Accumulator<Dirichlet>, Accumulator<Vector>,
                                      SettableTo<DirichletEstimator>, ICloneable
    {
        /// <summary>
        /// Number of samples
        /// </summary>
        public int N;

        /// <summary>
        /// Sum
        /// </summary>
        public Vector Sum;

        /// <summary>
        /// Sum of squares
        /// </summary>
        public Vector Sum2;

        /// <summary>
        /// Dimension of the Dirichlet
        /// </summary>
        public int Dimension
        {
            get { return Sum.Count; }
        }

        /// <summary>
        /// Creates a new Dirichlet estimator
        /// </summary>
        /// <param name="dimension">Dimension</param>
        public DirichletEstimator(int dimension)
        {
            Sum = Vector.Zero(dimension);
            Sum2 = Vector.Zero(dimension);
        }

        /// <summary>
        /// Gets the estimated distribution
        /// </summary>
        /// <param name="result">Where to put the estimated distribution</param>
        /// <returns>The estimated distribution</returns>
        public Dirichlet GetDistribution(Dirichlet result)
        {
            if (result == null) result = Dirichlet.Uniform(Dimension);
            if (N == 0) result.SetToUniform();
            else
            {
                double scale = 1.0/N;
                result.SetMeanAndMeanSquare(Sum*scale, Sum2*scale);
            }
            return result;
        }

        /// <summary>
        /// Adds a Dirichlet item to the estimator
        /// </summary>
        /// <param name="item">A Dirichlet instance</param>
        public void Add(Dirichlet item)
        {
            Vector m = item.GetMean();
            Sum.SetToSum(Sum, m);
            Vector m2 = m*m + item.GetVariance();
            Sum2.SetToSum(Sum2, m2);
            N++;
        }

        /// <summary>
        /// Adds a Vector sample to the estimator
        /// </summary>
        /// <param name="item"></param>
        public void Add(Vector item)
        {
            Sum.SetToSum(Sum, item);
            Sum2.SetToSum(Sum2, item*item);
            N++;
        }

        /// <summary>
        /// Clears the estimator
        /// </summary>
        public void Clear()
        {
            N = 0;
            Sum.SetAllElementsTo(0.0);
            Sum2.SetAllElementsTo(0.0);
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(DirichletEstimator value)
        {
            N = value.N;
            Sum.SetTo(value.Sum);
            Sum2.SetTo(value.Sum2);
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            DirichletEstimator result = new DirichletEstimator(Dimension);
            result.SetTo(this);
            return result;
        }
    }
}