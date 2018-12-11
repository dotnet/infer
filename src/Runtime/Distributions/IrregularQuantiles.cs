using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Microsoft.ML.Probabilistic.Distributions
{
    public class IrregularQuantiles : CanGetQuantile, CanGetProbLessThan
    {
        /// <summary>
        /// Sorted in ascending order.
        /// </summary>
        private readonly double[] probabilities, quantiles;

        public IrregularQuantiles(double[] probabilities, double[] quantiles)
        {
            AssertIncreasing(probabilities, nameof(probabilities));
            AssertInRange(probabilities, nameof(probabilities));
            AssertNondecreasing(quantiles, nameof(quantiles));
            AssertFinite(quantiles, nameof(quantiles));
            this.probabilities = probabilities;
            this.quantiles = quantiles;
        }

        private void AssertNondecreasing(double[] array, string paramName)
        {
            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] < array[i - 1]) throw new ArgumentException($"Invalid array: [{i}] {array[i]} < [{i - 1}] {array[i - 1]}", paramName);
            }
        }

        internal static void AssertFinite(double[] array, string paramName)
        {
            for (int i = 0; i < array.Length; i++)
            {
                if (double.IsInfinity(array[i])) throw new ArgumentException($"Array element is infinite: [{i}] {array[i]}", paramName);
                if (double.IsNaN(array[i])) throw new ArgumentException($"Array element is NaN: [{i}] {array[i]}", paramName);
            }
        }

        private void AssertIncreasing(double[] array, string paramName)
        {
            for (int i = 1; i < array.Length; i++)
            {
                if (array[i] <= array[i - 1]) throw new ArgumentException($"Invalid array: [{i}] {array[i]} <= [{i - 1}] {array[i - 1]}", paramName);
            }
        }

        internal static void AssertInRange(double[] array, string paramName)
        {
            for (int i = 0; i < array.Length; i++)
            {
                if (array[i] < 0) throw new ArgumentException($"Array[{i}] {array[i]} < 0", paramName);
                if (array[i] > 1) throw new ArgumentException($"Array[{i}] {array[i]} > 1", paramName);
                if (double.IsNaN(array[i])) throw new ArgumentException($"Array element is NaN: [{i}] {array[i]}", paramName);
            }
        }

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
                double slope = (quantiles[largerIndex] - quantiles[smallerIndex]) / (probabilities[largerIndex] - probabilities[smallerIndex]);
                return quantiles[smallerIndex] + (probability - probabilities[smallerIndex]) * slope;
            }
        }

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
                double slope = (probabilities[largerIndex] - probabilities[smallerIndex]) / (quantiles[largerIndex] - quantiles[smallerIndex]);
                return probabilities[smallerIndex] + (x - quantiles[smallerIndex]) * slope;
            }
        }
    }
}
