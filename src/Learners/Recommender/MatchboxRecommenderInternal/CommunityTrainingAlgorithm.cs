// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.MatchboxRecommenderInternal
{
    using System.Diagnostics;

    using GaussianArray = Distributions.DistributionStructArray<Distributions.Gaussian, double>;
    using GaussianMatrix = Distributions.DistributionRefArray<Distributions.DistributionStructArray<Distributions.Gaussian, double>, double[]>;

    /// <summary>
    /// Represents a Matchbox recommender algorithm which learns parameters of multiple users, items and features.
    /// </summary>
    internal class CommunityTrainingAlgorithm
    {
        /// <summary>
        /// The training algorithm.
        /// </summary>
        private readonly MatchboxCommunityTraining_EP inferenceAlgorithm;

        /// <summary>
        /// The number of inference iterations to perform.
        /// </summary>
        private readonly int iterationCount;

        /// <summary>
        /// Initializes a new instance of the <see cref="CommunityTrainingAlgorithm"/> class.
        /// </summary>
        /// <param name="iterationCount">The number of inference iterations to perform.</param>
        /// <param name="traitCount">The number of traits the algorithm will try to learn.</param>
        /// <param name="useSharedUserThresholds">Indicates whether to use shared user thresholds.</param>
        /// <param name="noiseHyperparameters">The noise-related hyper-parameters.</param>
        /// <param name="userHyperparameters">The user-related hyper-parameters.</param>
        /// <param name="itemHyperparameters">The item-related hyper-parameters.</param>
        /// <param name="userFeatureHyperparameters">The user feature related hyper-parameters.</param>
        /// <param name="itemFeatureHyperparameters">The item feature related hyper-parameters.</param>
        public CommunityTrainingAlgorithm(
            int iterationCount,
            int traitCount,
            bool useSharedUserThresholds,
            NoiseHyperparameters noiseHyperparameters,
            UserHyperparameters userHyperparameters,
            ItemHyperparameters itemHyperparameters,
            FeatureHyperparameters userFeatureHyperparameters,
            FeatureHyperparameters itemFeatureHyperparameters)
        {
            this.inferenceAlgorithm = new MatchboxCommunityTraining_EP();

            Debug.Assert(iterationCount > 0, "The number of iterations must be positive.");
            this.iterationCount = iterationCount;

            Debug.Assert(traitCount > 0, "The number of traits must be positive.");
            this.inferenceAlgorithm.TraitCount = traitCount;

            this.inferenceAlgorithm.UseSharedUserThresholds = useSharedUserThresholds;

            this.SetNoiseHyperparameters(noiseHyperparameters);
            this.SetUserHyperparameters(userHyperparameters);
            this.SetItemHyperparameters(itemHyperparameters);
            this.SetUserFeatureHyperparameters(userFeatureHyperparameters);
            this.SetItemFeatureHyperparameters(itemFeatureHyperparameters);
        }

        /// <summary>
        /// Sets the observed values of the user and item metadata - both the features and their corresponding counts.
        /// </summary>
        /// <param name="metadata">The metadata to observe.</param>
        public void SetObservedMetadata(InstanceMetadata metadata)
        {
            Debug.Assert(
                metadata.UserCount > 0 && metadata.ItemCount > 0, 
                "The training set must contain at least one user and item.");
            Debug.Assert(
                metadata.UserFeatures.EntityCount == metadata.UserCount, 
                "Features should be provided for all users in the training set, and only for them.");
            Debug.Assert(
                metadata.ItemFeatures.EntityCount == metadata.ItemCount, 
                "Features should be provided for all items in the training set, and only for them.");

            var thresholdCount = metadata.RatingCount + 1;
            this.inferenceAlgorithm.UserCount = metadata.UserCount;
            this.inferenceAlgorithm.ItemCount = metadata.ItemCount;
            this.inferenceAlgorithm.UserThresholdCount = thresholdCount;
            this.inferenceAlgorithm.MiddleUserThresholdIndex = thresholdCount / 2;
            this.inferenceAlgorithm.UserThresholdPriorMean = AlgorithmUtils.GetUserThresholdPriorMeans(thresholdCount);
            this.inferenceAlgorithm.UserFeatureCount = metadata.UserFeatures.FeatureCount;
            this.inferenceAlgorithm.ItemFeatureCount = metadata.ItemFeatures.FeatureCount;
            this.inferenceAlgorithm.NonZeroUserFeatureValues = metadata.UserFeatures.NonZeroFeatureValues;
            this.inferenceAlgorithm.NonZeroUserFeatureIndices = metadata.UserFeatures.NonZeroFeatureIndices;
            this.inferenceAlgorithm.NonZeroUserFeatureCounts = metadata.UserFeatures.NonZeroFeatureCounts;
            this.inferenceAlgorithm.NonZeroItemFeatureValues = metadata.ItemFeatures.NonZeroFeatureValues;
            this.inferenceAlgorithm.NonZeroItemFeatureIndices = metadata.ItemFeatures.NonZeroFeatureIndices;
            this.inferenceAlgorithm.NonZeroItemFeatureCounts = metadata.ItemFeatures.NonZeroFeatureCounts;
        }

        /// <summary>
        /// Sets the observed instance data.
        /// </summary>
        /// <param name="instanceData">The instance data to observe.</param>
        public void SetObservedInstanceData(InstanceData instanceData)
        {
            this.inferenceAlgorithm.ObservationCount = instanceData.UserIds.Count;

            this.inferenceAlgorithm.UserIds = instanceData.UserIds;
            this.inferenceAlgorithm.ItemIds = instanceData.ItemIds;
            this.inferenceAlgorithm.Ratings = instanceData.Ratings;
        }

        /// <summary>
        /// Constrains the per-user and per-item model parameters.
        /// </summary>
        /// <param name="constraints">The parameters to constrain to.</param>
        /// <remarks>
        /// The per-entity parameters are constrained to their corresponding output messages
        /// (or more concretely, to the marginal divided by the prior). This is required in
        /// order to carry message information over several batches of the data.
        /// </remarks>
        public void ConstrainEntityParameters(ParameterDistributions constraints)
        {
            this.inferenceAlgorithm.UserTraitsMessage = constraints.UserTraitDistribution;
            this.inferenceAlgorithm.UserBiasMessage = constraints.UserBiasDistribution;
            this.inferenceAlgorithm.UserThresholdsMessage = constraints.UserThresholdDistribution;
            this.inferenceAlgorithm.ItemTraitsMessage = constraints.ItemTraitDistribution;
            this.inferenceAlgorithm.ItemBiasMessage = constraints.ItemBiasDistribution;
        }

        /// <summary>
        /// Initializes the per-user and per-item model parameters.
        /// </summary>
        /// <param name="initializers">The parameters to initialize to.</param>
        /// <remarks>
        /// The per-entity parameter initialization is required for correct reconstruction of the 
        /// intermediate messages by the special first iteration. The initializers have to contain 
        /// the corresponding posteriors from the last iteration. Note that feature-related parameters 
        /// do not need to be initialized, because messages to the features are not reconstructed by 
        /// the special first iteration.
        /// </remarks>
        public void InitializeEntityParameters(ParameterDistributions initializers)
        {
            this.inferenceAlgorithm.UserTraitsInitializer = initializers.UserTraitDistribution;
            this.inferenceAlgorithm.UserBiasInitializer = initializers.UserBiasDistribution;
            this.inferenceAlgorithm.UserThresholdsInitializer = initializers.UserThresholdDistribution;
            this.inferenceAlgorithm.ItemTraitsInitializer = initializers.ItemTraitDistribution;
            this.inferenceAlgorithm.ItemBiasInitializer = initializers.ItemBiasDistribution;
        }

        /// <summary>
        /// Infers the parameters of the Matchbox model.
        /// </summary>
        /// <param name="inferFeatures">Indicates whether user and item features should be inferred.</param>
        /// <returns>Distributions over the parameters of the Matchbox model inferred from the data.</returns>
        /// <remarks>This method can be called only after all observed values in the model have been set.</remarks>
        public ParameterDistributions InferParameters(bool inferFeatures)
        {
            this.inferenceAlgorithm.Execute(this.iterationCount);

            GaussianMatrix userTraitFeatureWeights = null;
            GaussianArray userBiasFeatureWeights = null;
            GaussianMatrix itemTraitFeatureWeights = null;
            GaussianArray itemBiasFeatureWeights = null;

            if (inferFeatures)
            {
                userTraitFeatureWeights = this.inferenceAlgorithm.UserTraitFeatureWeightsMarginal();
                userBiasFeatureWeights = this.inferenceAlgorithm.UserBiasFeatureWeightsMarginal();
                itemTraitFeatureWeights = this.inferenceAlgorithm.ItemTraitFeatureWeightsMarginal();
                itemBiasFeatureWeights = this.inferenceAlgorithm.ItemBiasFeatureWeightsMarginal();
            }

            return new ParameterDistributions(
                this.inferenceAlgorithm.UserTraitsMarginal(),
                this.inferenceAlgorithm.UserBiasMarginal(),
                this.inferenceAlgorithm.UserThresholdsMarginal(),
                this.inferenceAlgorithm.ItemTraitsMarginal(),
                this.inferenceAlgorithm.ItemBiasMarginal(),
                userTraitFeatureWeights,
                userBiasFeatureWeights,
                itemTraitFeatureWeights,
                itemBiasFeatureWeights);
        }

        /// <summary>
        /// Gets the output messages of the parameters of the Matchbox model.
        /// </summary>
        /// <returns>The output messages.</returns>
        /// <remarks>
        /// This method does not run inference.
        /// This method can be called only after all observed values in the model have been set.
        /// </remarks>
        public ParameterDistributions GetOutputMessages()
        {
            return new ParameterDistributions(
                this.inferenceAlgorithm.UserTraitsMarginalDividedByPrior(),
                this.inferenceAlgorithm.UserBiasMarginalDividedByPrior(),
                this.inferenceAlgorithm.UserThresholdsMarginalDividedByPrior(),
                this.inferenceAlgorithm.ItemTraitsMarginalDividedByPrior(),
                this.inferenceAlgorithm.ItemBiasMarginalDividedByPrior(),
                null,
                null,
                null,
                null);
        }

        #region Helper methods

        /// <summary>
        /// Sets the noise hyper-parameters.
        /// </summary>
        /// <param name="noiseHyperparameters">The noise hyper-parameters.</param>
        private void SetNoiseHyperparameters(NoiseHyperparameters noiseHyperparameters)
        {
            Debug.Assert(noiseHyperparameters != null, "Valid noise hyperparameters must be provided.");

            this.inferenceAlgorithm.AffinityNoiseVariance = noiseHyperparameters.AffinityVariance;
            this.inferenceAlgorithm.UserThresholdNoiseVariance = noiseHyperparameters.UserThresholdVariance;
        }

        /// <summary>
        /// Sets the user hyper-parameters.
        /// </summary>
        /// <param name="userHyperparameters">The user hyper-parameters.</param>
        private void SetUserHyperparameters(UserHyperparameters userHyperparameters)
        {
            Debug.Assert(userHyperparameters != null, "Valid noise hyperparameters must be provided.");

            this.inferenceAlgorithm.UserTraitVariance = userHyperparameters.TraitVariance;
            this.inferenceAlgorithm.UserBiasVariance = userHyperparameters.BiasVariance;
            this.inferenceAlgorithm.UserThresholdPriorVariance = userHyperparameters.ThresholdPriorVariance;
        }

        /// <summary>
        /// Sets the item hyper-parameters.
        /// </summary>
        /// <param name="itemHyperparameters">The item hyper-parameters.</param>
        private void SetItemHyperparameters(ItemHyperparameters itemHyperparameters)
        {
            Debug.Assert(itemHyperparameters != null, "Valid item hyperparameters must be provided.");

            this.inferenceAlgorithm.ItemTraitVariance = itemHyperparameters.TraitVariance;
            this.inferenceAlgorithm.ItemBiasVariance = itemHyperparameters.BiasVariance;
        }

        /// <summary>
        /// Sets the user feature related hyper-parameters.
        /// </summary>
        /// <param name="userFeatureHyperparameters">The user feature related hyper-parameters.</param>
        private void SetUserFeatureHyperparameters(FeatureHyperparameters userFeatureHyperparameters)
        {
            Debug.Assert(userFeatureHyperparameters != null, "Valid user feature hyperparameters must be provided.");

            this.inferenceAlgorithm.UserTraitFeatureWeightPriorVariance = userFeatureHyperparameters.TraitWeightPriorVariance;
            this.inferenceAlgorithm.UserBiasFeatureWeightPriorVariance = userFeatureHyperparameters.BiasWeightPriorVariance;
        }

        /// <summary>
        /// Sets the item feature related hyper-parameters for a given inference algorithm.
        /// </summary>
        /// <param name="itemFeatureHyperparameters">The user feature related hyper-parameters.</param>
        private void SetItemFeatureHyperparameters(FeatureHyperparameters itemFeatureHyperparameters)
        {
            Debug.Assert(itemFeatureHyperparameters != null, "Valid item feature hyperparameters must be provided.");

            this.inferenceAlgorithm.ItemTraitFeatureWeightPriorVariance = itemFeatureHyperparameters.TraitWeightPriorVariance;
            this.inferenceAlgorithm.ItemBiasFeatureWeightPriorVariance = itemFeatureHyperparameters.BiasWeightPriorVariance;
        }

        #endregion
    }
}
