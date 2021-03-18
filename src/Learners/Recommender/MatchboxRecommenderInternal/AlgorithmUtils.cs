// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Linq;

    using Microsoft.ML.Probabilistic.Distributions;
    using Microsoft.ML.Probabilistic.Factors;

    using GaussianArray = Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// Provides utility functions for the Matchbox algorithm.
    /// </summary>
    internal static class AlgorithmUtils
    {
        /// <summary>
        /// Gets the array of prior means for user thresholds given their number.
        /// </summary>
        /// <param name="userThresholdCount">The number of user thresholds.</param>
        /// <returns>The array of prior means for user thresholds.</returns>
        public static double[] GetUserThresholdPriorMeans(int userThresholdCount)
        {
            Debug.Assert(userThresholdCount >= 0, "The number of user thresholds must be non-negative.");

            var result = new double[userThresholdCount];
            result[0] = double.NegativeInfinity;
            result[userThresholdCount - 1] = double.PositiveInfinity;
            for (int i = 1; i < userThresholdCount - 1; ++i)
            {
                result[i] = i - (userThresholdCount / 2.0) + 0.5;
            }

            return result;
        }

        /// <summary>
        /// Gets the parameter distributions of the average user.
        /// </summary>
        /// <param name="parameterDistributions">The learned posterior parameter distributions.</param>
        /// <param name="userFeatures">The user features.</param>
        /// <returns>The average user parameter distributions.</returns>
        public static UserParameterDistribution GetAverageUserParameters(
            ParameterDistributions parameterDistributions, SparseFeatureMatrix userFeatures)
        {
            GaussianArray traitDistribution;
            Gaussian biasDistribution;
            GaussianArray thresholdDistribution = GetAverageParameterVector(parameterDistributions.UserThresholdDistribution);

            GetAverageTraitsAndBiases(
                parameterDistributions.UserTraitDistribution,
                parameterDistributions.UserBiasDistribution,
                parameterDistributions.UserFeature.TraitWeights,
                parameterDistributions.UserFeature.BiasWeights,
                userFeatures,
                out traitDistribution,
                out biasDistribution);

            return new UserParameterDistribution(traitDistribution, biasDistribution, thresholdDistribution);
        }

        /// <summary>
        /// Gets the parameter distributions of the average item.
        /// </summary>
        /// <param name="parameterDistributions">The learned posterior parameter distributions.</param>
        /// <param name="itemFeatures">The item features.</param>
        /// <returns>The average item parameter distributions.</returns>
        public static ItemParameterDistribution GetAverageItemParameters(
            ParameterDistributions parameterDistributions, SparseFeatureMatrix itemFeatures)
        {
            GaussianArray traitDistribution;
            Gaussian biasDistribution;

            GetAverageTraitsAndBiases(
                parameterDistributions.ItemTraitDistribution,
                parameterDistributions.ItemBiasDistribution,
                parameterDistributions.ItemFeature.TraitWeights,
                parameterDistributions.ItemFeature.BiasWeights,
                itemFeatures,
                out traitDistribution,
                out biasDistribution);

            return new ItemParameterDistribution(traitDistribution, biasDistribution);
        }

        /// <summary>
        /// Adds the feature contribution for a given user/item to its traits and bias.
        /// </summary>
        /// <param name="parameters">The user/item parameters.</param>
        /// <param name="featureWeights">The feature weights.</param>
        /// <param name="featureValues">The feature values.</param>
        /// <param name="adjustedTraits">The resulting traits.</param>
        /// <param name="adjustedBias">The resulting bias.</param>
        public static void AddFeatureContribution(
            EntityParameterDistribution parameters,
            FeatureParameterDistribution featureWeights,
            SparseFeatureVector featureValues,
            out GaussianArray adjustedTraits,
            out Gaussian adjustedBias)
        {
            adjustedTraits = new GaussianArray(
                parameters.Traits.Count,
                i => DoublePlusOp.SumAverageConditional(
                    parameters.Traits[i],
                    ComputeFeatureContribution(featureWeights.TraitWeights[i], featureValues.NonZeroFeatureValues, featureValues.NonZeroFeatureIndices)));

            adjustedBias = DoublePlusOp.SumAverageConditional(
                parameters.Bias,
                ComputeFeatureContribution(featureWeights.BiasWeights, featureValues.NonZeroFeatureValues, featureValues.NonZeroFeatureIndices));
        }

        #region Helper methods

        /// <summary>
        /// Gets the average traits and biases amongst all users or items.
        /// </summary>
        /// <param name="traits">The user/item traits.</param>
        /// <param name="biases">The user/item biases.</param>
        /// <param name="traitFeatureWeights">The feature weights for the traits.</param>
        /// <param name="biasFeatureWeights">The feature weights for the biases.</param>
        /// <param name="featureValues">The feature values of all users/items.</param>
        /// <param name="averageTraits">The resulting average traits.</param>
        /// <param name="averageBias">The resulting average bias.</param>
        /// <remarks>
        /// When features are used, this method will exclude the feature contribution when computing the average.
        /// Therefore, the feature contribution for a given user/item has to be then added at prediction time.
        /// </remarks>
        private static void GetAverageTraitsAndBiases(
            GaussianMatrix traits,
            GaussianArray biases,
            GaussianMatrix traitFeatureWeights,
            GaussianArray biasFeatureWeights,
            SparseFeatureMatrix featureValues,
            out GaussianArray averageTraits,
            out Gaussian averageBias)
        {
            if (featureValues.FeatureCount == 0)
            {
                averageTraits = GetAverageParameterVector(traits);
                averageBias = GetAverageParameter(biases);
            }
            else
            {
                averageTraits = GetAverageParameterVectorExcludingFeatureContribution(traits, traitFeatureWeights, featureValues);
                averageBias = GetAverageParameterExcludingFeatureContribution(biases, biasFeatureWeights, featureValues);
            }
        }

        /// <summary>
        /// Gets the average row of a parameter matrix.
        /// </summary>
        /// <param name="entityParameterVectors">
        /// The matrix of parameters.
        /// Each row of the matrix is expected to be the parameter vector of a given user or item.
        /// </param>
        /// <returns>The average user/item parameter vector.</returns>
        /// <remarks>
        /// Each element in the returned row is the average of all elements in the corresponding column.
        /// </remarks>
        private static GaussianArray GetAverageParameterVector(GaussianMatrix entityParameterVectors)
        {
            Debug.Assert(entityParameterVectors.Count > 0, "The number of entities must be positive.");
            return new GaussianArray(
                entityParameterVectors[0].Count, 
                i => GetAverageParameter(entityParameterVectors.Select(x => x[i])));
        }

        /// <summary>
        /// Gets the average row of a parameter matrix excluding the feature contribution.
        /// </summary>
        /// <param name="entityParameterVectors">
        /// The matrix of parameters.
        /// Each row of the matrix is expected to be the parameter vector of a given user or item.
        /// </param>
        /// <param name="weights">The feature weights.</param>
        /// <param name="features">The feature values.</param>
        /// <returns>The average user/item parameter vector excluding the feature contribution.</returns>
        private static GaussianArray GetAverageParameterVectorExcludingFeatureContribution(
            GaussianMatrix entityParameterVectors, GaussianMatrix weights, SparseFeatureMatrix features)
        {
            Debug.Assert(entityParameterVectors.Count > 0, "The number of entities must be positive.");
            return new GaussianArray(
                entityParameterVectors[0].Count,
                i => GetAverageParameterExcludingFeatureContribution(entityParameterVectors.Select(x => x[i]), weights[i], features));
        }

        /// <summary>
        /// Gets the average parameter from an array of parameters.
        /// </summary>
        /// <param name="entityParameters">The array of parameters.</param>
        /// <returns>The average parameter.</returns>
        /// <remarks>
        /// Point masses are excluded from this computation. 
        /// Only infinities make an exception because they are used in the user thresholds.
        /// </remarks>
        private static Gaussian GetAverageParameter(IEnumerable<Gaussian> entityParameters)
        {
            var filteredEntityParameters = entityParameters.Where(x => !x.IsPointMass || double.IsInfinity(x.GetMean())).ToArray();
            return Distribution.SetToProductOfAll(new Gaussian(), filteredEntityParameters);
        }

        /// <summary>
        /// Computes the average parameter of an array of user/item parameters 
        /// excluding the feature contribution for the users or items.
        /// </summary>
        /// <param name="entityParameters">The array of user/item parameters to get the average of.</param>
        /// <param name="weights">The feature weights of each user/item.</param>
        /// <param name="features">The feature values of each user/item.</param>
        /// <returns>The average parameter distribution excluding feature contributions.</returns>
        private static Gaussian GetAverageParameterExcludingFeatureContribution(
            IEnumerable<Gaussian> entityParameters,
            IReadOnlyList<Gaussian> weights,
            SparseFeatureMatrix features)
        {
            var adjustedEntityParameters = entityParameters.Select((gaussian, i) => 
                DoublePlusOp.AAverageConditional(gaussian, ComputeFeatureContribution(weights, features.NonZeroFeatureValues[i], features.NonZeroFeatureIndices[i])));

            return GetAverageParameter(adjustedEntityParameters);
        }

        /// <summary>
        /// Computes the inner product of the feature weights and values.
        /// </summary>
        /// <param name="weights">The feature weights.</param>
        /// <param name="nonZeroValues">The sparse feature values.</param>
        /// <param name="nonZeroIndices">The sparse feature indices.</param>
        /// <returns>The contribution of the features.</returns>
        private static Gaussian ComputeFeatureContribution(IReadOnlyList<Gaussian> weights, IReadOnlyList<double> nonZeroValues, IReadOnlyList<int> nonZeroIndices)
        {
            Debug.Assert(nonZeroValues.Count == nonZeroIndices.Count, "The number of values must be equal to the number of indices.");
            int count = nonZeroValues.Count;

            var nonZeroWeights = new Gaussian[count];
            SubarrayOp<double>.ItemsAverageConditional(weights, nonZeroIndices, nonZeroWeights);

            var products = new List<Gaussian>(count);
            for (int i = 0; i < count; ++i)
            {
                products.Add(GaussianProductOp.ProductAverageConditional(nonZeroWeights[i], nonZeroValues[i]));
            }

            return FastSumOp.SumAverageConditional(products);
        }

        #endregion
    }
}
