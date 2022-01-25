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
    using System.Threading.Tasks;
    using Microsoft.ML.Probabilistic.Collections;
    using Microsoft.ML.Probabilistic.Math;
    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a distribution using the quantiles at probabilities (0,...,n-1)/(n-1)
    /// </summary>
    [Serializable, DataContract]
    public class OuterQuantiles : CanGetQuantile<double>, CanGetProbLessThan<double>, CanGetLogProb<double>
    {
        /// <summary>
        /// Numbers in increasing order.
        /// </summary>
        [DataMember]
        private readonly double[] quantiles;

        public OuterQuantiles(double[] quantiles)
        {
            if (quantiles == null) throw new ArgumentNullException(nameof(quantiles));
            if (quantiles.Length == 0) throw new ArgumentException("quantiles array is empty", nameof(quantiles));
            AssertNondecreasing(quantiles, nameof(quantiles));
            AssertFinite(quantiles, nameof(quantiles));
            this.quantiles = quantiles;
        }

        /// <inheritdoc/>
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
            return $"OuterQuantiles({quantiles.Length}, {quantileString})";
        }

        public double[] ToArray()
        {
            return quantiles;
        }

        /// <inheritdoc/>
        public override bool Equals(object obj)
        {
            if (!(obj is OuterQuantiles that)) return false;
            return quantiles.ValueEquals(that.quantiles);
        }

        /// <inheritdoc/>
        public override int GetHashCode()
        {
            return Hash.GetHashCodeAsSequence(quantiles);
        }

        internal static void AssertFinite(double[] array, string paramName)
        {
            for (int i = 0; i < array.Length; i++)
            {
                if (double.IsInfinity(array[i])) throw new ArgumentOutOfRangeException(paramName, array[i], $"Array element is infinite: {paramName}[{i}]={array[i]}");
                if (double.IsNaN(array[i])) throw new ArgumentOutOfRangeException(paramName, array[i], $"Array element is NaN: {paramName}[{i}]={array[i]}");
            }
        }

        internal static void AssertNondecreasing(double[] array, string paramName)
        {
            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] < array[i - 1]) throw new ArgumentException($"Decreasing array elements: {paramName}[{i}] {array[i]} < {paramName}[{i - 1}] {array[i - 1]}", paramName);
            }
        }

        public static OuterQuantiles FromDistribution(int quantileCount, CanGetQuantile<double> canGetQuantile)
        {
            var quantiles = Util.ArrayInit(quantileCount, i => canGetQuantile.GetQuantile(i / (quantileCount - 1.0)));
            return new OuterQuantiles(quantiles);
        }

        /// <inheritdoc/>
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
            if (n == 0) throw new ArgumentException("quantiles array is empty", nameof(quantiles));
            // The index of the specified value in the specified array, if value is found; otherwise, a negative number. 
            // If value is not found and value is less than one or more elements in array, the negative number returned 
            // is the bitwise complement of the index of the first element that is larger than value. 
            // If value is not found and value is greater than all elements in array, the negative number returned is 
            // the bitwise complement of (the index of the last element plus 1). 
            int index = Array.BinarySearch(quantiles, x);
            if (index >= 0)
            {
                while (index > 0 && quantiles[index - 1] == quantiles[index]) index--;
                if (index == 0) return 0;
                else return (double)index / (n - 1);
            }
            index = ~index;
            if (index == 0) return 0.0;
            if (index >= n) return 1.0;
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
            return (index - 1 + frac) / (n - 1);
        }

        /// <inheritdoc/>
        public double GetLogProb(double value)
        {
            int n = quantiles.Length;
            if (n == 0) throw new ArgumentException("quantiles array is empty", nameof(quantiles));
            // The index of the specified value in the specified array, if value is found; otherwise, a negative number. 
            // If value is not found and value is less than one or more elements in array, the negative number returned 
            // is the bitwise complement of the index of the first element that is larger than value. 
            // If value is not found and value is greater than all elements in array, the negative number returned is 
            // the bitwise complement of (the index of the last element plus 1). 
            int index = Array.BinarySearch(quantiles, value);
            if (index < 0)
            {
                index = ~index;
            }
            if (index == 0) index++;
            if (index >= n) index = n - 1;
            // quantiles[index-1] < x <= quantiles[index]
            double diff = quantiles[index] - quantiles[index - 1];
            return 1.0 / (n - 1) / Math.Max(1e-4, diff);
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound)
        {
            return Math.Max(0.0, GetProbLessThan(upperBound) - GetProbLessThan(lowerBound));
        }

        /// <inheritdoc/>
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
            if (quantiles == null) throw new ArgumentNullException(nameof(quantiles));
            int n = quantiles.Length;
            if (n == 0) throw new ArgumentException("quantiles array is empty", nameof(quantiles));
            if (probability == 1.0)
            {
                return MMath.NextDouble(quantiles[n - 1]);
            }
            if (n == 1) return quantiles[0];
            double pos = MMath.LargestDoubleProduct(probability, n - 1);
            int lower = (int)Math.Floor(pos);
            if (lower == n - 1) return quantiles[lower];
            return GetQuantile(probability, lower, quantiles[lower], quantiles[lower + 1], n);
        }

        /// <summary>
        /// Returns the largest quantile such that ((quantile - lowerItem)/(upperItem - lowerItem) + lowerIndex)/(n-1) &lt;= probability.
        /// </summary>
        /// <param name="probability">A number between 0 and 1.</param>
        /// <param name="lowerIndex"></param>
        /// <param name="lowerItem">Must be finite.</param>
        /// <param name="upperItem">Must be finite and at least lowerItem.</param>
        /// <param name="n">Must be greater than 1</param>
        /// <returns></returns>
        internal static double GetQuantile(double probability, double lowerIndex, double lowerItem, double upperItem, long n)
        {
            if(probability < 0) throw new ArgumentOutOfRangeException(nameof(probability), probability, $"{nameof(probability)} < 0");
            if (probability > 1) throw new ArgumentOutOfRangeException(nameof(probability), probability, $"{nameof(probability)} > 1");
            if (n <= 1) throw new ArgumentOutOfRangeException(nameof(n), n, "n <= 1");
            double pos = MMath.LargestDoubleProduct(probability, n - 1);
            double frac = MMath.LargestDoubleSum(-lowerIndex, pos);
            if (upperItem < lowerItem) throw new ArgumentOutOfRangeException(nameof(upperItem), upperItem, $"{nameof(upperItem)} ({upperItem}) < {nameof(lowerItem)} ({lowerItem})");
            if (upperItem == lowerItem) return lowerItem;
            // The above check ensures diff > 0
            double diff = upperItem - lowerItem;
            if (diff > double.MaxValue)
            {
                double scale = Math.Max(Math.Abs(upperItem), Math.Abs(lowerItem));
                double lowerItemOverScale = lowerItem / scale;
                diff = upperItem / scale - lowerItemOverScale;
                double offset = MMath.LargestDoubleProduct(frac, diff);
                return MMath.LargestDoubleSum(lowerItemOverScale, offset) * scale;
            }
            else
            {
                double offset = MMath.LargestDoubleProduct(frac, diff);
                return MMath.LargestDoubleSum(lowerItem, offset);
            }
        }
    }
}
