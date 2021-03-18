// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    /// <summary>
    /// Represents a sparse feature vector.
    /// </summary>
    internal class SparseFeatureVector
    {
        /// <summary>
        /// Prevents a default instance of the <see cref="SparseFeatureVector"/> class from being created.
        /// </summary>
        private SparseFeatureVector()
        {
        }

        /// <summary>
        /// Gets the array of non-zero feature values.
        /// </summary>
        public IReadOnlyList<double> NonZeroFeatureValues { get; private set; }

        /// <summary>
        /// Gets the array of non-zero feature indices.
        /// </summary>
        public IReadOnlyList<int> NonZeroFeatureIndices { get; private set; }

        /// <summary>
        /// Gets the total number of features.
        /// </summary>
        public int FeatureCount { get; private set; }

        /// <summary>
        /// Creates a new sparse feature vector referencing the specified arrays.
        /// </summary>
        /// <param name="nonZeroFeatureValues">The array of non-zero feature values.</param>
        /// <param name="nonZeroFeatureIndices">The array of non-zero feature indices.</param>
        /// <param name="featureCount">The total number of features in the vector.</param>
        /// <returns>The created sparse feature vector.</returns>
        public static SparseFeatureVector Create(IReadOnlyList<double> nonZeroFeatureValues, IReadOnlyList<int> nonZeroFeatureIndices, int featureCount)
        {
            Debug.Assert(CanBeCreatedFrom(nonZeroFeatureValues, nonZeroFeatureIndices, featureCount), "Invalid arguments provided.");

            return new SparseFeatureVector
            {
                NonZeroFeatureValues = nonZeroFeatureValues,
                NonZeroFeatureIndices = nonZeroFeatureIndices,
                FeatureCount = featureCount
            };
        }

        /// <summary>
        /// Creates a zero sparse feature vector with the specified number of features.
        /// </summary>
        /// <param name="featureCount">The total number of features in the vector.</param>
        /// <returns>The created sparse feature vector.</returns>
        public static SparseFeatureVector CreateAllZero(int featureCount)
        {
            Debug.Assert(featureCount >= 0, "Feature count can not be negative.");

            return new SparseFeatureVector
            {
                NonZeroFeatureValues = new double[0],
                NonZeroFeatureIndices = new int[0],
                FeatureCount = featureCount
            };
        }

        /// <summary>
        /// Checks if a sparse feature vector can be created from the specified arguments,
        /// i.e. they are valid and self-consistent.
        /// </summary>
        /// <param name="nonZeroFeatureValues">The array of non-zero feature values.</param>
        /// <param name="nonZeroFeatureIndices">The array of non-zero feature indices.</param>
        /// <param name="featureCount">The total number of features in the vector.</param>
        /// <returns>
        /// True if a sparse feature vector can be created from the specified arguments, false otherwise.
        /// </returns>
        public static bool CanBeCreatedFrom(
            IReadOnlyList<double> nonZeroFeatureValues, IReadOnlyList<int> nonZeroFeatureIndices, int featureCount)
        {
            if (featureCount < 0)
            {
                // Feature count can not be negative
                return false;
            }
            
            if (nonZeroFeatureValues == null || nonZeroFeatureIndices == null)
            {
                // Both arrays should be valid
                return false;
            }

            if (nonZeroFeatureValues.Count != nonZeroFeatureIndices.Count)
            {
                // Arrays of values and indices have different length
                return false;
            }

            if (nonZeroFeatureIndices.Any(index => index < 0 || index >= featureCount))
            {
                // Negative feature indices are not allowed
                return false;
            }

            if (nonZeroFeatureIndices.Distinct().Count() != nonZeroFeatureIndices.Count)
            {
                // There are duplicate indices
                return false;
            }
            
            return true;
        }
    }
}
