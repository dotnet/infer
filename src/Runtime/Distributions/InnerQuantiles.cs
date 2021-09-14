// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Runtime.Serialization;
    using System.Text;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a distribution using the quantiles at probabilities (1,...,n)/(n+1)
    /// </summary>
    [Serializable, DataContract]
    public class InnerQuantiles : CanGetQuantile<double>, CanGetProbLessThan<double>
    {
        /// <summary>
        /// Numbers in increasing order.  Cannot be empty.
        /// </summary>
        [DataMember] private readonly double[] quantiles;
        /// <summary>
        /// Gaussian approximation of the lower tail.
        /// </summary>
        private readonly Gaussian lowerGaussian;
        /// <summary>
        /// Gaussian approximation of the upper tail.
        /// </summary>
        private readonly Gaussian upperGaussian;

        public InnerQuantiles(double[] quantiles)
        {
            if (quantiles == null) throw new ArgumentNullException(nameof(quantiles));
            if (quantiles.Length == 0) throw new ArgumentException("quantiles array is empty", nameof(quantiles));
            OuterQuantiles.AssertFinite(quantiles, nameof(quantiles));
            OuterQuantiles.AssertNondecreasing(quantiles, nameof(quantiles));
            this.quantiles = quantiles;
            lowerGaussian = GetLowerGaussian(quantiles);
            upperGaussian = GetUpperGaussian(quantiles);
        }

        public static InnerQuantiles FromDistribution(int quantileCount, CanGetQuantile<double> canGetQuantile)
        {
            if (quantileCount == 0) throw new ArgumentOutOfRangeException(nameof(quantileCount), quantileCount, "quantileCount == 0");
            var quantiles = Util.ArrayInit(quantileCount, i => canGetQuantile.GetQuantile((i + 1.0) / (quantileCount + 1.0)));
            return new InnerQuantiles(quantiles);
        }

        public override string ToString()
        {
            string quantileString;
            if (quantiles.Length <= 5)
            {
                quantileString = StringUtil.CollectionToString(quantiles, " ");
            }
            else
            {
                int n = quantiles.Length;
                quantileString = $"{quantiles[0]:g2} {quantiles[1]:g2} ... {quantiles[n - 2]:g2} {quantiles[n - 1]:g2}";
            }
            return $"InnerQuantiles({quantiles.Length}, {quantileString})";
        }

        public double[] ToArray()
        {
            return quantiles;
        }

        public override bool Equals(object obj)
        {
            if (!(obj is InnerQuantiles that)) return false;
            return quantiles.ValueEquals(that.quantiles);
        }

        public override int GetHashCode()
        {
            return Hash.GetHashCodeAsSequence(quantiles);
        }

        /// <inheritdoc/>
        public double GetProbLessThan(double x)
        {
            int n = quantiles.Length;
            if (x < quantiles[0])
            {
                return Math.Min(lowerGaussian.GetProbLessThan(x), 1.0/(n+1));
            }
            if (x > quantiles[n - 1])
            {
                return Math.Max(upperGaussian.GetProbLessThan(x), n/(n+1.0));
            }
            return GetProbLessThan(x, quantiles);
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound)
        {
            return Math.Max(0.0, GetProbLessThan(upperBound) - GetProbLessThan(lowerBound));
        }

        /// <summary>
        /// Get the quantile rank of x.
        /// </summary>
        /// <param name="x">Any real number.</param>
        /// <param name="quantiles">Cutpoints, sorted in increasing order, corresponding to probability i/(n+1)</param>
        /// <returns>A real number in [0,1]</returns>
        public static double GetProbLessThan(double x, double[] quantiles)
        {
            if (double.IsNaN(x)) throw new ArgumentOutOfRangeException("x is NaN");
            int n = quantiles.Length;
            if (n == 0) throw new ArgumentOutOfRangeException(nameof(quantiles) + ".Count", n, "quantiles array is empty");
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
            double diff = quantiles[index] - quantiles[index - 1];
            double frac;
            if (diff > double.MaxValue)
            {
                double scale = Math.Max(Math.Abs(quantiles[index]), Math.Abs(quantiles[index - 1]));
                double q = quantiles[index - 1] / scale;
                frac = (x / scale - q) / (quantiles[index] / scale - q);
            }
            else
            {
                frac = (x - quantiles[index - 1]) / diff;
            }
            return (index + frac) / (n + 1);
        }

        private static Gaussian GetLowerGaussian(IReadOnlyList<double> quantiles)
        {
            int n = quantiles.Count;
            if (n == 0) throw new ArgumentOutOfRangeException(nameof(quantiles) + ".Count", n, "quantiles array is empty");
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
                Gaussian result = Gaussian.FromMeanAndVariance(mean, stddev * stddev);
                if (result.IsPointMass || !result.IsProper() || double.IsNaN(result.GetMean()) || double.IsInfinity(result.GetMean()))
                {
                    return Gaussian.PointMass(quantiles[0]);
                }
                return result;
            }
        }

        private static Gaussian GetUpperGaussian(IReadOnlyList<double> quantiles)
        {
            int n = quantiles.Count;
            if (n == 0) throw new ArgumentOutOfRangeException(nameof(quantiles) + ".Count", n, "quantiles array is empty");
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
                GetGaussianFromQuantiles(quantiles[n - 1], p0, quantiles[i], p1, out mean, out stddev);
                if (double.IsNaN(mean) || double.IsInfinity(mean))
                {
                    return Gaussian.PointMass(quantiles[n - 1]);
                }
                Gaussian result = Gaussian.FromMeanAndVariance(mean, stddev * stddev);
                if (!result.IsProper() || double.IsNaN(result.GetMean()) || double.IsInfinity(result.GetMean()))
                {
                    return Gaussian.PointMass(quantiles[n - 1]);
                }
                return result;
            }
        }

        private static void GetGaussianFromQuantiles(double x0, double p0, double x1, double p1, out double mean, out double deviation)
        {
            // solve for the Gaussian mean and stddev that yield:
            // x0 = mean + stddev * NormalCdfInv(p0)
            double z0 = MMath.NormalCdfInv(p0);
            double z1 = MMath.NormalCdfInv(p1);
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
        public static double GetProbLessThan(double x, IReadOnlyList<double> quantiles)
        {
            int n = quantiles.Count;
            if (n == 0) throw new ArgumentOutOfRangeException(nameof(quantiles) + ".Count", n, "quantiles array is empty");
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
            double diff = quantiles[index] - quantiles[index - 1];
            double frac;
            if (diff > double.MaxValue)
            {
                double scale = Math.Max(Math.Abs(quantiles[index]), Math.Abs(quantiles[index - 1]));
                double q = quantiles[index - 1] / scale;
                frac = (x / scale - q) / (quantiles[index] / scale - q);
            }
            else
            {
                frac = (x - quantiles[index - 1]) / diff;
            }
            return (index + frac) / (n + 1);
        }

        /// <summary>
        /// Returns the largest value x such that GetProbLessThan(x) &lt;= probability.
        /// </summary>
        /// <param name="probability">A real number in [0,1].</param>
        /// <returns></returns>
        public double GetQuantile(double probability)
        {
            if (probability < 0) throw new ArgumentOutOfRangeException(nameof(probability), "probability < 0");
            if (probability > 1.0) throw new ArgumentOutOfRangeException(nameof(probability), "probability > 1.0");
            int n = quantiles.Length;
            if (probability < 1.0 / (n + 1.0))
            {
                return Math.Min(quantiles[0], lowerGaussian.GetQuantile(probability));
            }
            if (probability > n / (n + 1.0))
            {
                return Math.Max(quantiles[n-1], upperGaussian.GetQuantile(probability));
            }
            if (n == 1) return quantiles[0]; // probability must be 0.5
            double pos = MMath.LargestDoubleProduct(probability, n + 1) - 1;
            int lower = (int)Math.Floor(pos);
            if (lower == n - 1) return quantiles[lower];
            return OuterQuantiles.GetQuantile(probability, lower + 1, quantiles[lower], quantiles[lower + 1], n + 2);
        }
    }
}
