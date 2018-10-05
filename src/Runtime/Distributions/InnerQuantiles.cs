// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a distribution using the quantiles at probabilities (1,...,n)/(n+1)
    /// </summary>
    public class InnerQuantiles : CanGetQuantile, CanGetProbLessThan
    {
        /// <summary>
        /// Numbers in increasing order.
        /// </summary>
        private readonly double[] quantiles;
        /// <summary>
        /// Gaussian approximation of the lower tail.
        /// </summary>
        Gaussian lowerGaussian;
        /// <summary>
        /// Gaussian approximation of the upper tail.
        /// </summary>
        Gaussian upperGaussian;

        private InnerQuantiles(int quantileCount)
        {
            this.quantiles = new double[quantileCount];
            CacheGaussians();
        }

        public InnerQuantiles(int quantileCount, CanGetQuantile canGetQuantile) : this(quantileCount)
        {
            for (int i = 0; i < quantileCount; i++)
            {
                this.quantiles[i] = canGetQuantile.GetQuantile((i + 1.0) / (quantileCount + 1.0));
            }
            CacheGaussians();
        }

        public override string ToString()
        {
            string quantileString;
            if(quantiles.Length <= 5)
            {
                quantileString = StringUtil.CollectionToString(quantiles, " ");
            }
            else
            {
                int n = quantiles.Length;
                quantileString = $"{quantiles[0]:g2} {quantiles[1]:g2} ... {quantiles[n-2]:g2} {quantiles[n-1]:g2}";
            }
            return $"InnerQuantiles({quantiles.Length}, {quantileString})";
        }

        public double[] ToArray()
        {
            return quantiles;
        }

        public double GetProbLessThan(double x)
        {
            if (x < quantiles[0])
            {
                return lowerGaussian.GetProbLessThan(x);
            }
            int n = quantiles.Length;
            if (x > quantiles[n - 1])
            {
                return upperGaussian.GetProbLessThan(x);
            }
            return GetProbLessThan(x, quantiles);
        }

        private static void GetGaussianFromQuantiles(double x0, double p0, double x1, double p1, out double mean, out double deviation)
        {
            double z0 = MMath.NormalCdfInv(p0);
            double z1 = MMath.NormalCdfInv(p1);
            // solve for the equivalent Gaussian mean and stddev
            Matrix Z = new Matrix(new double[,] { { 1, z0 }, { 1, z1 } });
            DenseVector X = DenseVector.FromArray(x0, x1);
            DenseVector A = DenseVector.Zero(2);
            A.SetToLeastSquares(X, Z);
            mean = A[0];
            deviation = A[1];
        }

        /// <summary>
        /// Get the quantile rank of x.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="quantiles">Cutpoints, sorted in increasing order, corresponding to probability i/(n+1)</param>
        /// <returns>A real number in [0,1]</returns>
        public static double GetProbLessThan(double x, double[] quantiles)
        {
            int n = quantiles.Length;
            if (x < quantiles[0])
            {
                return GetProbLessThan(x, (IReadOnlyList<double>)quantiles);
            }
            if (x > quantiles[n - 1])
            {
                return GetProbLessThan(x, (IReadOnlyList<double>)quantiles);
            }
            // The index of the specified value in the specified array, if value is found; otherwise, a negative number. 
            // If value is not found and value is less than one or more elements in array, the negative number returned 
            // is the bitwise complement of the index of the first element that is larger than value. 
            // If value is not found and value is greater than all elements in array, the negative number returned is 
            // the bitwise complement of (the index of the last element plus 1). 
            int index = Array.BinarySearch(quantiles, x);
            if (index >= 0)
            {
                while (index > 0 && quantiles[index - 1] == quantiles[index]) index--;
                return (index + 1.0) / (n + 1);
            }
            index = ~index;
            // quantiles[index-1] < x < quantiles[index]
            double frac = (x - quantiles[index - 1]) / (quantiles[index] - quantiles[index - 1]);
            return (index + frac) / (n + 1);
        }

        private void CacheGaussians()
        {
            lowerGaussian = GetLowerGaussian(quantiles);
            upperGaussian = GetUpperGaussian(quantiles);
        }

        private static Gaussian GetLowerGaussian(IReadOnlyList<double> quantiles)
        {
            int n = quantiles.Count;
            // find the next quantile
            int i = 1;
            while (i < n && quantiles[i] == quantiles[0])
            {
                i++;
            }
            if (i == n)
            {
                // all quantiles are equal
                return Gaussian.PointMass(quantiles[0]);
            }
            else
            {
                double p0 = 1.0 / (n + 1);
                double p1 = (i + 1.0) / (n + 1);
                double mean, stddev;
                GetGaussianFromQuantiles(quantiles[0], p0, quantiles[i], p1, out mean, out stddev);
                return Gaussian.FromMeanAndVariance(mean, stddev * stddev);
            }
        }

        private static Gaussian GetUpperGaussian(IReadOnlyList<double> quantiles)
        {
            int n = quantiles.Count;
            // find the previous quantile
            int i = n - 2;
            while (i >= 0 && quantiles[i] == quantiles[n - 1])
            {
                i--;
            }
            if (i < 0)
            {
                // all quantiles are equal
                return Gaussian.PointMass(quantiles[n - 1]);
            }
            else
            {
                double p0 = (double)n / (n + 1);
                double p1 = (i + 1.0) / (n + 1);
                double mean, stddev;
                GetGaussianFromQuantiles(quantiles[n-1], p0, quantiles[i], p1, out mean, out stddev);
                return Gaussian.FromMeanAndVariance(mean, stddev * stddev);
            }
        }

        /// <summary>
        /// Get the quantile rank of x.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="quantiles">Cutpoints, sorted in increasing order, corresponding to probability i/(n+1)</param>
        /// <returns>A real number in [0,1]</returns>
        public static double GetProbLessThan(double x, IReadOnlyList<double> quantiles)
        {
            int n = quantiles.Count;
            if (x < quantiles[0])
            {
                return GetLowerGaussian(quantiles).GetProbLessThan(x);
            }
            if (x > quantiles[n - 1])
            {
                return GetUpperGaussian(quantiles).GetProbLessThan(x);
            }
            // IReadOnlyList does not have BinarySearch.
            // Start at 1 because we already know that x > quantiles[0].
            int index = 1;
            while (x > quantiles[index])
            {
                index++;
            }
            // quantiles[index-1] < x <= quantiles[index]
            double frac = (x - quantiles[index - 1]) / (quantiles[index] - quantiles[index - 1]);
            return (index + frac) / (n + 1);
        }

        public double GetQuantile(double probability)
        {
            int n = quantiles.Length;
            if(probability < 1.0/(n+1.0))
            {
                return lowerGaussian.GetQuantile(probability);
            }
            if(probability > n/(n+1.0))
            {
                return upperGaussian.GetQuantile(probability);
            }
            double pos = probability * (quantiles.Length + 1) - 1;
            int lower = (int)Math.Floor(pos);
            int upper = (int)Math.Ceiling(pos);
            if (upper == lower) upper = lower + 1;
            return OuterQuantiles.GetQuantile(probability, lower + 1, quantiles[lower], quantiles[upper], quantiles.Length + 2);
        }
    }
}
