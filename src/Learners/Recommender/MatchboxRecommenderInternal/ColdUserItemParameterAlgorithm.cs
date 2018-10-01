// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System.Diagnostics;

    using Microsoft.ML.Probabilistic.Distributions;

    using GaussianArray = Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<Microsoft.ML.Probabilistic.Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// Represents a Matchbox recommender algorithm which infers the parameters of cold users and items.
    /// </summary>
    internal class ColdUserItemParameterAlgorithm
    {
        /// <summary>
        /// The posteriors over the user feature related parameters learned during community training.
        /// </summary>
        private readonly FeatureParameterDistribution userFeatureParameterPosteriors;

        /// <summary>
        /// The posteriors over the item feature related parameters learned during community training.
        /// </summary>
        private readonly FeatureParameterDistribution itemFeatureParameterPosteriors;

        /// <summary>
        /// The prior distribution of a cold user excluding the effect of features.
        /// This is computed as the average from all training users.
        /// </summary>
        private readonly UserParameterDistribution userParameterDistributionAverage;

        /// <summary>
        /// The prior distribution of a cold item excluding the effect of features.
        /// This is computed as the average from all training items.
        /// </summary>
        private readonly ItemParameterDistribution itemParameterDistributionAverage;

        /// <summary>
        /// Initializes a new instance of the <see cref="ColdUserItemParameterAlgorithm"/> class.
        /// </summary>
        /// <param name="userFeatureParameterPosteriors">The posteriors over the user feature related parameters learned during community training.</param>
        /// <param name="itemFeatureParameterPosteriors">The posteriors over the item feature related parameters learned during community training.</param>
        /// <param name="userParameterDistributionAverage">The average trait, bias, and threshold posterior over all users in training.</param>
        /// <param name="itemParameterDistributionAverage">The average trait and bias posterior over all items in training.</param>
        public ColdUserItemParameterAlgorithm(
            FeatureParameterDistribution userFeatureParameterPosteriors,
            FeatureParameterDistribution itemFeatureParameterPosteriors,
            UserParameterDistribution userParameterDistributionAverage,
            ItemParameterDistribution itemParameterDistributionAverage)
        {
            this.userFeatureParameterPosteriors = userFeatureParameterPosteriors;
            this.itemFeatureParameterPosteriors = itemFeatureParameterPosteriors;
            this.userParameterDistributionAverage = userParameterDistributionAverage;
            this.itemParameterDistributionAverage = itemParameterDistributionAverage;
        }

        /// <summary>
        /// Infers parameters for a given cold user.
        /// </summary>
        /// <param name="userFeatures">The user features.</param>
        /// <returns>A distribution over the user parameters.</returns>
        public UserParameterDistribution InferUserParameters(SparseFeatureVector userFeatures)
        {
            UserParameterDistribution result;

            if (this.userFeatureParameterPosteriors.FeatureCount == 0)
            {
                Debug.Assert(
                    userFeatures.FeatureCount == 0,
                    "The number of user features passed must be equal to the number of user features learned.");

                result = this.userParameterDistributionAverage;
            }
            else
            {
                GaussianArray traits;
                Gaussian bias;

                AlgorithmUtils.AddFeatureContribution(
                    this.userParameterDistributionAverage,
                    this.userFeatureParameterPosteriors,
                    userFeatures,
                    out traits,
                    out bias);

                result = new UserParameterDistribution(traits, bias, this.userParameterDistributionAverage.Thresholds);
            }

            return result;
        }

        /// <summary>
        /// Infers parameters for a given cold item.
        /// </summary>
        /// <param name="itemFeatures">The item features.</param>
        /// <returns>A distribution over the item parameters.</returns>
        public ItemParameterDistribution InferItemParameters(SparseFeatureVector itemFeatures)
        {
            ItemParameterDistribution result;

            if (this.itemFeatureParameterPosteriors.FeatureCount == 0)
            {
                Debug.Assert(
                    itemFeatures.FeatureCount == 0,
                    "The number of item features passed must be equal to the number of item features learned.");

                result = this.itemParameterDistributionAverage;
            }
            else
            {
                GaussianArray traits;
                Gaussian bias;

                AlgorithmUtils.AddFeatureContribution(
                    this.itemParameterDistributionAverage,
                    this.itemFeatureParameterPosteriors,
                    itemFeatures,
                    out traits,
                    out bias);                

                result = new ItemParameterDistribution(traits, bias);
            }

            return result;
        }
    }
}
