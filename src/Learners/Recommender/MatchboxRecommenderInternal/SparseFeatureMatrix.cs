// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Utilities;

    /// <summary>
    /// Represents a sparse feature matrix.
    /// </summary>
    internal class SparseFeatureMatrix
    {
        /// <summary>
        /// Prevents a default instance of the <see cref="SparseFeatureMatrix"/> class from being created.
        /// </summary>
        private SparseFeatureMatrix()
        {
        }

        /// <summary>
        /// Gets the total number of features.
        /// </summary>
        public int FeatureCount { get; private set; }

        /// <summary>
        /// Gets the array of arrays of non-zero feature values for each entity.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<double>> NonZeroFeatureValues { get; private set; }

        /// <summary>
        /// Gets the array of arrays of non-zero feature indices for each entity.
        /// </summary>
        public IReadOnlyList<IReadOnlyList<int>> NonZeroFeatureIndices { get; private set; }

        /// <summary>
        /// Gets the array non-zero feature counts for each entity.
        /// </summary>
        public IReadOnlyList<int> NonZeroFeatureCounts { get; private set; }

        /// <summary>
        /// Gets the number of entities.
        /// </summary>
        public int EntityCount { get; private set; }

        /// <summary>
        /// Creates a new sparse feature matrix referencing the specified arrays.
        /// </summary>
        /// <param name="featureCount">The number of features.</param>
        /// <param name="nonZeroFeatureValues">The array of arrays of non-zero feature values for each entity.</param>
        /// <param name="nonZeroFeatureIndices">The array of arrays of non-zero feature indices for each entity.</param>
        /// <returns>The created sparse feature matrix.</returns>
        public static SparseFeatureMatrix Create(
            int featureCount,
            IReadOnlyList<IReadOnlyList<double>> nonZeroFeatureValues,
            IReadOnlyList<IReadOnlyList<int>> nonZeroFeatureIndices)
        {
            Debug.Assert(CanBeCreatedFrom(nonZeroFeatureValues, nonZeroFeatureIndices), "Invalid arguments provided.");

            return new SparseFeatureMatrix
            {
                FeatureCount = featureCount,
                NonZeroFeatureValues = nonZeroFeatureValues,
                NonZeroFeatureIndices = nonZeroFeatureIndices,
                EntityCount = nonZeroFeatureValues.Count,
                NonZeroFeatureCounts = Util.ArrayInit(nonZeroFeatureValues.Count, i => nonZeroFeatureIndices[i].Count)
            };
        }

        /// <summary>
        /// Creates a zero sparse feature matrix for the given number of entities.
        /// </summary>
        /// <param name="entityCount">The number of entities in the matrix.</param>
        /// <returns>The created sparse feature matrix.</returns>
        public static SparseFeatureMatrix CreateAllZero(int entityCount)
        {
            Debug.Assert(entityCount >= 0, "Entity count can not be negative.");

            var emptyDoubleArray = new double[0];
            var emptyIntArray = new int[0];

            return new SparseFeatureMatrix
            {
                NonZeroFeatureValues = Util.ArrayInit(entityCount, i => emptyDoubleArray),
                NonZeroFeatureIndices = Util.ArrayInit(entityCount, i => emptyIntArray),
                FeatureCount = 0,
                EntityCount = entityCount,
                NonZeroFeatureCounts = Util.ArrayInit(entityCount, i => 0)
            };
        }

        /// <summary>
        /// Checks if a sparse feature matrix can be created from the specified arguments,
        /// i.e. they are valid and self-consistent.
        /// </summary>
        /// <param name="nonZeroFeatureValues">The array of arrays of non-zero feature values for each entity.</param>
        /// <param name="nonZeroFeatureIndices">The array of arrays of non-zero feature indices for each entity.</param>
        /// <returns>
        /// True if a sparse feature matrix can be created from the specified arguments, false otherwise.
        /// </returns>
        public static bool CanBeCreatedFrom(
            IReadOnlyList<IReadOnlyList<double>> nonZeroFeatureValues,
            IReadOnlyList<IReadOnlyList<int>> nonZeroFeatureIndices)
        {
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

            for (int i = 0; i < nonZeroFeatureValues.Count; ++i)
            {
                if (nonZeroFeatureValues[i] == null || nonZeroFeatureIndices[i] == null)
                {
                    // One of the inner arrays is null
                    return false;
                }

                if (nonZeroFeatureValues[i].Count != nonZeroFeatureIndices[i].Count)
                {
                    // The corresponding inner arrays have different length
                    return false;
                }

                if (nonZeroFeatureIndices[i].Any(index => index < 0))
                {
                    // Negative feature indices are not allowed
                    return false;
                }

                if (nonZeroFeatureIndices[i].Distinct().Count() != nonZeroFeatureIndices[i].Count)
                {
                    // There are duplicate indices
                    return false;
                }
            }

            return true;
        }
    }
}
