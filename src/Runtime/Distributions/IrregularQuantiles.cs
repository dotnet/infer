// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System;
using Microsoft.ML.Probabilistic.Math;

namespace Microsoft.ML.Probabilistic.Distributions
{
    /// <summary>
    /// Represents a distribution using the quantiles at arbitrary probabilities.
    /// </summary>
    public class IrregularQuantiles : CanGetQuantile<double>, CanGetProbLessThan<double>
    {
        /// <summary>
        /// Sorted in ascending order.
        /// </summary>
        private readonly double[] probabilities, quantiles;

        public IrregularQuantiles(double[] probabilities, double[] quantiles)
        {
            AssertIncreasing(probabilities, nameof(probabilities));
            AssertInRange(probabilities, nameof(probabilities));
            OuterQuantiles.AssertNondecreasing(quantiles, nameof(quantiles));
            OuterQuantiles.AssertFinite(quantiles, nameof(quantiles));
            this.probabilities = probabilities;
            this.quantiles = quantiles;
        }

        private void AssertIncreasing(double[] array, string paramName)
        {
            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] <= array[i - 1]) throw new ArgumentException($"Array is not increasing: {paramName}[{i}] {array[i]} <= {paramName}[{i - 1}] {array[i - 1]}", paramName);
            }
        }

        private static void AssertInRange(double[] array, string paramName)
        {
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] < 0) throw new ArgumentOutOfRangeException(paramName, $"{paramName}[{i}] {array[i]} < 0");
                if (array[i] > 1) throw new ArgumentOutOfRangeException(paramName, $"{paramName}[{i}] {array[i]} > 1");
                if (double.IsNaN(array[i])) throw new ArgumentOutOfRangeException(paramName, $"{paramName}[{i}] {array[i]}");
            }
        }

        /// <summary>
        /// Returns the quantile rank of x.  This is a probability such that GetQuantile(probability) == x, whenever x is inside the support of the distribution.  May be discontinuous due to duplicates.
        /// </summary>
        /// <param name="x"></param>
        /// <returns>A real number in [0,1]</returns>
        public double GetProbLessThan(double x)
        {
            int index = Array.BinarySearch(quantiles, x);
            if (index >= 0)
            {
                // In case of duplicates, find the smallest copy.
                while (index > 0 && quantiles[index - 1] == x) index--;
                return probabilities[index];
            }
            else
            {
                // Linear interpolation
                int largerIndex = ~index;
                if (largerIndex == 0) return 0;
                if (largerIndex == quantiles.Length) return 1;
                int smallerIndex = largerIndex - 1;
                double upperItem = quantiles[largerIndex];
                double lowerItem = quantiles[smallerIndex];
                double diff = upperItem - lowerItem;
                double frac;
                if (diff > double.MaxValue)
                {
                    double scale = System.Math.Max(System.Math.Abs(upperItem), System.Math.Abs(lowerItem));
                    double lowerItemOverScale = lowerItem / scale;
                    diff = upperItem / scale - lowerItemOverScale;
                    double slope = diff / (probabilities[largerIndex] - probabilities[smallerIndex]);
                    frac = (x / scale - lowerItemOverScale) / slope;
                }
                else
                {
                    double slope = diff / (probabilities[largerIndex] - probabilities[smallerIndex]);
                    frac = (x - lowerItem) / slope;
                }
                return probabilities[smallerIndex] + frac;
            }
        }

        /// <inheritdoc/>
        public double GetProbBetween(double lowerBound, double upperBound)
        {
            return System.Math.Max(0.0, GetProbLessThan(upperBound) - GetProbLessThan(lowerBound));
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
            // The zero-based index of item in the sorted List<T>, if item is found; 
            // otherwise, a negative number that is the bitwise complement of the index of the next element that is larger than item 
            // or, if there is no larger element, the bitwise complement of Count.
            int index = Array.BinarySearch(probabilities, probability);
            if(index >= 0)
            {
                return quantiles[index];
            }
            else
            {
                // Linear interpolation
                int largerIndex = ~index;
                if (largerIndex == 0) return quantiles[largerIndex];
                int smallerIndex = largerIndex - 1;
                if (largerIndex == probabilities.Length) return quantiles[smallerIndex];
                double upperItem = quantiles[largerIndex];
                double lowerItem = quantiles[smallerIndex];
                double diff = upperItem - lowerItem;
                if (diff > double.MaxValue)
                {
                    double scale = System.Math.Max(System.Math.Abs(upperItem), System.Math.Abs(lowerItem));
                    double lowerItemOverScale = lowerItem / scale;
                    diff = upperItem / scale - lowerItemOverScale;
                    double slope = diff / (probabilities[largerIndex] - probabilities[smallerIndex]);
                    double frac = MMath.LargestDoubleSum(-probabilities[smallerIndex], probability);
                    double offset = MMath.LargestDoubleProduct(frac, slope);
                    return MMath.LargestDoubleSum(lowerItemOverScale, offset) * scale;
                }
                else
                {
                    double slope = diff / (probabilities[largerIndex] - probabilities[smallerIndex]);
                    // Solve for the largest x such that probabilities[smallerIndex] + (x - quantiles[smallerIndex]) / slope <= probability.
                    double frac = MMath.LargestDoubleSum(-probabilities[smallerIndex], probability);
                    double offset = MMath.LargestDoubleProduct(frac, slope);
                    return MMath.LargestDoubleSum(lowerItem, offset);
                }
            }
        }
    }
}
