// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System.Diagnostics;

    using Microsoft.ML.Probabilistic.Distributions;

    using GaussianArray = Distributions.DistributionStructArray<Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// Represents a Matchbox recommender algorithm which predicts ratings for a given user and an item.
    /// </summary>
    internal class RatingPredictionAlgorithm
    {
        /// <summary>
        /// The rating prediction algorithm.
        /// </summary>
        private readonly MatchboxRatingPrediction_EP inferenceAlgorithm;

        /// <summary>
        /// Initializes a new instance of the <see cref="RatingPredictionAlgorithm"/> class.
        /// </summary>
        /// <param name="noiseHyperparameters">The noise hyper-parameters.</param>
        public RatingPredictionAlgorithm(NoiseHyperparameters noiseHyperparameters)
        {
            this.inferenceAlgorithm = new MatchboxRatingPrediction_EP();
            this.SetNoiseHyperparameters(noiseHyperparameters);
        }

        /// <summary>
        /// Infers the distribution over the rating which a given user will give to an item.
        /// </summary>
        /// <param name="userParameterPosteriors">The posteriors over user parameters.</param>
        /// <param name="itemParameterPosteriors">The posteriors over item parameters.</param>
        /// <returns>The distribution over the rating.</returns>
        public Discrete InferRatingDistribution(
            UserParameterDistribution userParameterPosteriors, ItemParameterDistribution itemParameterPosteriors)
        {
            Debug.Assert(userParameterPosteriors != null && itemParameterPosteriors != null, "A valid posteriors must be provided.");
            Debug.Assert(
                userParameterPosteriors.Traits.Count == itemParameterPosteriors.Traits.Count,
                "Given posteriors should have the same associated number of traits.");

            this.inferenceAlgorithm.TraitCount = userParameterPosteriors.Traits.Count;

            this.SetupUserFromPosteriors(userParameterPosteriors);
            this.SetupItemFromPosteriors(itemParameterPosteriors);

            this.inferenceAlgorithm.ObservationCount = 1;
            this.inferenceAlgorithm.UserIds = new[] { 0 };
            this.inferenceAlgorithm.ItemIds = new[] { 0 };

            this.inferenceAlgorithm.Execute(1);

            return this.inferenceAlgorithm.RatingsMarginal()[0];
        }

        #region Helper methods

        /// <summary>
        /// Sets up noise hyper-parameters.
        /// </summary>
        /// <param name="noiseHyperparameters">The noise hyper-parameters.</param>
        private void SetNoiseHyperparameters(NoiseHyperparameters noiseHyperparameters)
        {
            Debug.Assert(noiseHyperparameters != null, "Valid noise hyperparameters must be provided.");

            this.inferenceAlgorithm.AffinityNoiseVariance = noiseHyperparameters.AffinityVariance;
            this.inferenceAlgorithm.UserThresholdNoiseVariance = noiseHyperparameters.UserThresholdVariance;
        }

        /// <summary>
        /// Sets up the inference algorithm to operate on a single user given the posteriors over the parameters of the user.
        /// </summary>
        /// <param name="userParameterPosteriors">The posteriors over the parameters of the user.</param>
        private void SetupUserFromPosteriors(UserParameterDistribution userParameterPosteriors)
        {
            Debug.Assert(userParameterPosteriors != null, "Valid user parameter posteriors must be provided.");

            this.inferenceAlgorithm.UserCount = 1;
            this.inferenceAlgorithm.UserThresholdCount = userParameterPosteriors.Thresholds.Count;
            this.inferenceAlgorithm.UserTraitsPrior = new GaussianMatrix(new[] { userParameterPosteriors.Traits });
            this.inferenceAlgorithm.UserBiasPrior = new GaussianArray(new[] { userParameterPosteriors.Bias });
            this.inferenceAlgorithm.UserThresholdsPrior = new GaussianMatrix(new[] { userParameterPosteriors.Thresholds });
        }

        /// <summary>
        /// Sets up the inference algorithm to operate on a single item given the posteriors over the parameters of the item.
        /// </summary>
        /// <param name="itemParameterPosteriors">The posteriors over the parameters of the item.</param>
        private void SetupItemFromPosteriors(ItemParameterDistribution itemParameterPosteriors)
        {
            Debug.Assert(itemParameterPosteriors != null, "Valid item parameter posteriors must be provided.");

            this.inferenceAlgorithm.ItemCount = 1;
            this.inferenceAlgorithm.ItemTraitsPrior = new GaussianMatrix(new[] { itemParameterPosteriors.Traits });
            this.inferenceAlgorithm.ItemBiasPrior = new GaussianArray(new[] { itemParameterPosteriors.Bias });
        }

        #endregion
    }
}
