// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Distributions
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Text;
    using System.Threading.Tasks;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// Represents a distribution using the quantiles at probabilities (0,...,n-1)/(n-1)
    /// </summary>
    public class OuterQuantiles : CanGetQuantile, CanGetProbLessThan
    {
        /// <summary>
        /// Numbers in increasing order.
        /// </summary>
        private readonly double[] quantiles;

        public OuterQuantiles(double[] quantiles)
        {
            AssertNondecreasing(quantiles, nameof(quantiles));
            AssertFinite(quantiles, nameof(quantiles));
            this.quantiles = quantiles;
        }

        internal static void AssertFinite(double[] array, string paramName)
        {
            for (int i = 0; i < array.Length; i++)
            {
                if (double.IsInfinity(array[i])) throw new ArgumentException($"Array element is infinite: [{i}] {array[i]}", paramName);
                if (double.IsNaN(array[i])) throw new ArgumentException($"Array element is NaN: [{i}] {array[i]}", paramName);
            }
        }

        internal static void AssertNondecreasing(double[] array, string paramName)
        {
            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] < array[i - 1]) throw new ArgumentException($"Invalid array: [{i}] {array[i]} < [{i - 1}] {array[i - 1]}", paramName);
            }
        }

        public OuterQuantiles(int quantileCount, CanGetQuantile canGetQuantile)
        {
            this.quantiles = new double[quantileCount];
            for (int i = 0; i < quantileCount; i++)
            {
                this.quantiles[i] = canGetQuantile.GetQuantile(i / (quantileCount - 1.0));
            }
        }

        public double GetProbLessThan(double x)
        {
            return GetProbLessThan(x, this.quantiles);
        }

        /// <summary>
        /// Returns the quantile rank of x.  This is a probability such that GetQuantile(probability) == x, whenever x is inside the support of the distribution.  May be discontinuous due to duplicates.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="quantiles">Quantiles, sorted in increasing order, corresponding to probabilities (0,...,n-1)/(n-1)</param>
        /// <returns>A real number in [0,1]</returns>
        public static double GetProbLessThan(double x, double[] quantiles)
        {
            int n = quantiles.Length;
            // The index of the specified value in the specified array, if value is found; otherwise, a negative number. 
            // If value is not found and value is less than one or more elements in array, the negative number returned 
            // is the bitwise complement of the index of the first element that is larger than value. 
            // If value is not found and value is greater than all elements in array, the negative number returned is 
            // the bitwise complement of (the index of the last element plus 1). 
            int index = Array.BinarySearch(quantiles, x);
            if (index >= 0)
            {
                while (index > 0 && quantiles[index - 1] == quantiles[index]) index--;
                return (double)index / (n - 1);
            }
            index = ~index;
            if (index == 0) return 0.0;
            if (index >= n) return 1.0;
            // quantiles[index-1] < x <= quantiles[index]
            double frac = (x - quantiles[index - 1]) / (quantiles[index] - quantiles[index - 1]);
            return (index - 1 + frac) / (n - 1);
        }

        /// <summary>
        /// Returns the largest value x such that GetProbLessThan(x) &lt;= probability.
        /// </summary>
        /// <param name="probability">A real number in [0,1].</param>
        /// <returns></returns>
        public double GetQuantile(double probability)
        {
            return GetQuantile(probability, this.quantiles);
        }

        /// <summary>
        /// Returns the largest value x such that GetProbLessThan(x) &lt;= probability.
        /// </summary>
        /// <param name="probability">A real number in [0,1].</param>
        /// <param name="quantiles">Numbers in increasing order corresponding to the quantiles for probabilities (0,...,n-1)/(n-1).</param>
        /// <returns></returns>
        public static double GetQuantile(double probability, double[] quantiles)
        {
            if (probability < 0) throw new ArgumentOutOfRangeException(nameof(probability), "probability < 0");
            if (probability > 1.0) throw new ArgumentOutOfRangeException(nameof(probability), "probability > 1.0");
            int n = quantiles.Length;
            if (quantiles == null) throw new ArgumentNullException(nameof(quantiles));
            if (n == 0) throw new ArgumentException("quantiles array is empty", nameof(quantiles));
            if (probability == 1.0)
            {
                return MMath.NextDouble(quantiles[n - 1]);
            }
            if (n == 1) return quantiles[0];
            double pos = MMath.LargestDoubleProduct(n - 1, probability);
            int lower = (int)Math.Floor(pos);
            if (lower == n - 1) return quantiles[lower];
            return GetQuantile(probability, lower, quantiles[lower], quantiles[lower + 1], n);
        }

        /// <summary>
        /// Returns the largest quantile such that ((quantile - lowerItem)/(upperItem - lowerItem) + lowerIndex)/(n-1) &lt;= probability.
        /// </summary>
        /// <param name="probability"></param>
        /// <param name="lowerIndex"></param>
        /// <param name="lowerItem"></param>
        /// <param name="upperItem"></param>
        /// <param name="n"></param>
        /// <returns></returns>
        internal static double GetQuantile(double probability, double lowerIndex, double lowerItem, double upperItem, long n)
        {
            if (n == 1) throw new ArgumentOutOfRangeException("n == 1");
            double pos = MMath.LargestDoubleProduct(n - 1, probability);
            double frac = MMath.LargestDoubleSum(-lowerIndex, pos);
            double diff = upperItem - lowerItem;
            if (diff == 0) return lowerItem;
            double offset = MMath.LargestDoubleProduct(diff, frac);
            return MMath.LargestDoubleSum(lowerItem, offset);
        }
    }
}
