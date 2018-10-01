// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Class for accumulating weighted noisy matrix observations,
    /// and computing sample count, mean matrix, and covariance matrix
    /// </summary>
    public class MatrixMeanVarianceAccumulator : SettableTo<MatrixMeanVarianceAccumulator>, ICloneable
    {
        /// <summary>
        /// Mean matrix
        /// </summary>
        public Matrix Mean;

        /// <summary>
        /// Covariance matrix
        /// </summary>
        public PositiveDefiniteMatrix Variance;

        /// <summary>
        /// Count
        /// </summary>
        public double Count;

        /// <summary>
        /// Temporary workspace
        /// </summary>
        private Matrix diff;

        /// <summary>
        /// The number of rows in the matrix
        /// </summary>
        public int Rows
        {
            get { return Mean.Rows; }
        }

        /// <summary>
        /// The number of columns in the matrix
        /// </summary>
        public int Cols
        {
            get { return Mean.Cols; }
        }

        /// <summary>
        /// Adds an observation
        /// </summary>
        /// <param name="x"></param>
        public void Add(Matrix x)
        {
            Add(x, 1);
        }

        /// <summary>
        /// Adds a weighted observation
        /// </summary>
        /// <param name="x"></param>
        /// <param name="weight"></param>
        public void Add(Matrix x, double weight)
        {
            // new_mean = mean + w/(count + w)*(x - mean)
            // new_var = count/(count + w)*(var + w/(count+w)*(x-mean)*(x-mean)')
            Count += weight;
            if (Count == 0) return;
            diff.SetToDifference(x, Mean);
            double s = weight/Count;
            Mean.SetToSum(1, Mean, s, diff);
            diff.SetToElementwiseProduct(diff, diff);
            Variance.SetToSum(1 - s, Variance, s*(1 - s), diff);
        }

        /// <summary>
        /// Adds a noisy observation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="noiseVariance"></param>
        /// <param name="weight"></param>
        /// <remarks>The contents of noiseVariance are modified.</remarks>
        public void Add(Matrix x, PositiveDefiniteMatrix noiseVariance, double weight)
        {
            // new_cov = count/(count + w)*(cov + w/(count+w)*(x-mean)*(x-mean)') + w/(count+w)*noiseVariance
            Add(x, weight);
            if (Count == 0) return;
            double s = weight/Count;
            Variance.SetToSum(1, Variance, s, noiseVariance);
        }

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        public void Clear()
        {
            Count = 0;
            Mean.SetAllElementsTo(0);
            Variance.SetAllElementsTo(0);
        }

        /// <summary>
        /// Constructs an accumulator for matrix observations
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        public MatrixMeanVarianceAccumulator(int rows, int cols)
        {
            Mean = new Matrix(rows, cols);
            Variance = new PositiveDefiniteMatrix(rows, cols);
            diff = new Matrix(rows, cols);
        }

        /// <summary>
        /// Sets the state of this estimator from the specified estimator.
        /// </summary>
        /// <param name="value"></param>
        public void SetTo(MatrixMeanVarianceAccumulator value)
        {
            Mean.SetTo(value.Mean);
            Variance.SetTo(value.Variance);
            Count = value.Count;
        }

        /// <summary>
        /// Returns a clone of this estimator.
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            MatrixMeanVarianceAccumulator result = new MatrixMeanVarianceAccumulator(Rows, Cols);
            result.SetTo(this);
            return result;
        }
    }
}