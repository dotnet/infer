// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;

namespace Microsoft.ML.Probabilistic.Math
{
    /// <summary>
    /// Class for accumulating weighted noisy vector observations,
    /// and computing sample count, mean vector, and covariance matrix
    /// </summary>
    public class VectorMeanVarianceAccumulator : SettableTo<VectorMeanVarianceAccumulator>, ICloneable
    {
        private Vector mean;
        private PositiveDefiniteMatrix cov;
        private double count;

        /// <summary>
        /// Temporary workspace
        /// </summary>
        private Vector diff;

        /// <summary>
        /// Count
        /// </summary>
        public double Count
        {
            get { return count; }
        }

        /// <summary>
        /// Mean
        /// </summary>
        public Vector Mean
        {
            get { return mean; }
        }

        /// <summary>
        /// Covariance
        /// </summary>
        public PositiveDefiniteMatrix Variance
        {
            get { return cov; }
        }

        /// <summary>
        /// The dimensionality of the vector
        /// </summary>
        public int Dimension
        {
            get { return mean.Count; }
        }

        /// <summary>
        /// Adds an observation 
        /// </summary>
        /// <param name="x"></param>
        public void Add(Vector x)
        {
            Add(x, 1);
        }

        /// <summary>
        /// Adds a weighted observation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="weight"></param>
        public void Add(Vector x, double weight)
        {
            // new_mean = mean + w/(count + w)*(x - mean)
            // new_cov = count/(count + w)*(cov + w/(count+w)*(x-mean)*(x-mean)')
            count += weight;
            if (count == 0) return;
            diff.SetToDifference(x, mean);
            double s = weight/count;
            mean.SetToSum(1.0, mean, s, diff);
            //outer.SetToOuter(diff,diff2);
            // this is more numerically stable:
            //outer.SetToOuter(diff, diff).Scale(s);
            cov.SetToSumWithOuter(cov, s, diff, diff);
            cov.Scale(1 - s);
        }

        /// <summary>
        /// Adds a noisy observation.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="noiseVariance"></param>
        /// <param name="weight"></param>
        public void Add(Vector x, PositiveDefiniteMatrix noiseVariance, double weight)
        {
            // new_cov = count/(count + w)*(cov + w/(count+w)*(x-mean)*(x-mean)') + w/(count+w)*noiseVariance
            Add(x, weight);
            if (count == 0) return;
            double s = weight/count;
            cov.SetToSum(1, cov, s, noiseVariance);
        }

        /// <summary>
        /// Clears the accumulator
        /// </summary>
        public void Clear()
        {
            count = 0;
            mean.SetAllElementsTo(0);
            cov.SetAllElementsTo(0);
        }

        /// <summary>
        /// Constructs an accumulator for vector observations
        /// </summary>
        /// <param name="dimension"></param>
        public VectorMeanVarianceAccumulator(int dimension)
        {
            mean = Vector.Zero(dimension);
            diff = Vector.Zero(dimension);
            cov = new PositiveDefiniteMatrix(dimension, dimension);
            // not needed: count = 0
        }

        public void SetTo(VectorMeanVarianceAccumulator value)
        {
            mean.SetTo(value.mean);
            cov.SetTo(value.cov);
            count = value.count;
        }

        public object Clone()
        {
            VectorMeanVarianceAccumulator result = new VectorMeanVarianceAccumulator(Dimension);
            result.SetTo(this);
            return result;
        }
    }
}