// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Diagnostics;
    using System.Xml.Serialization;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;

    /// <summary>
    /// The configuration element that describes a Matchbox model based recommender.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "MatchboxRecommender")]
    public class MatchboxRecommenderConfigurationElement : RecommenderConfigurationElement
    {
        /// <summary>
        /// Gets or sets the number of traits.
        /// </summary>
        public int? TraitCount { get; set; }

        /// <summary>
        /// Gets or sets the number of iterations.
        /// </summary>
        public int? IterationCount { get; set; }

        /// <summary>
        /// Gets or sets the number of batches.
        /// </summary>
        public int? BatchCount { get; set; }

        /// <summary>
        /// Gets or sets the flag indicating if user features should be used in the model.
        /// </summary>
        public bool? UseUserFeatures { get; set; }

        /// <summary>
        /// Gets or sets the flag indicating if item features should be used in the model.
        /// </summary>
        public bool? UseItemFeatures { get; set; }

        /// <summary>
        /// Gets or sets the flag indicating whether to use shared user thresholds.
        /// </summary>
        public bool? UseSharedUserThresholds { get; set; }

        /// <summary>
        /// Gets or sets the variance of the affinity noise.
        /// </summary>
        public double? AffinityNoiseVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the threshold noise.
        /// </summary>
        public double? ThresholdNoiseVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the traits.
        /// </summary>
        public double? TraitVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the bias.
        /// </summary>
        public double? BiasVariance { get; set; }

        /// <summary>
        /// Gets or sets the variance of the threshold prior.
        /// </summary>
        public double? ThresholdPriorVariance { get; set; }

        /// <summary>
        /// Gets or sets the rating loss function used by the recommender.
        /// </summary>
        public LossFunction? RatingLossFunction { get; set; }
        
        /// <summary>
        /// Creates an instance of the Matchbox-based recommender using the settings from this configuration element.
        /// </summary>
        /// <param name="mapping">The mapping for the recommender being created.</param>
        /// <returns>An instance of the Matchbox-based recommender.</returns>
        public override IRecommender<SplitInstanceSource<RecommenderDataset>, User, Item, int, RatingDistribution, DummyFeatureSource> Create(
            IStarRatingRecommenderMapping<SplitInstanceSource<RecommenderDataset>, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping)
        {
            var inferNetRecommender = MatchboxRecommender.Create(mapping);
            this.SetupMatchboxModel(inferNetRecommender.Settings);
            return inferNetRecommender;
        }

        /// <summary>
        /// Fills recommender settings using the settings from this configuration element.
        /// </summary>
        /// <param name="settings">The recommender settings to fill.</param>
        private void SetupMatchboxModel(MatchboxRecommenderSettings settings)
        {
            settings.Training.UseUserFeatures = this.UseUserFeatures.Value;
            settings.Training.UseItemFeatures = this.UseItemFeatures.Value;
            settings.Training.TraitCount = this.TraitCount.Value;
            settings.Training.BatchCount = this.BatchCount.Value;
            settings.Training.IterationCount = this.IterationCount.Value;
            settings.Training.UseSharedUserThresholds = this.UseSharedUserThresholds.Value;

            settings.Training.Advanced.AffinityNoiseVariance = this.AffinityNoiseVariance.Value;
            settings.Training.Advanced.UserThresholdNoiseVariance = this.ThresholdNoiseVariance.Value;
            settings.Training.Advanced.UserTraitVariance = this.TraitVariance.Value;
            settings.Training.Advanced.ItemTraitVariance = this.TraitVariance.Value;
            settings.Training.Advanced.UserBiasVariance = this.BiasVariance.Value;
            settings.Training.Advanced.ItemBiasVariance = this.BiasVariance.Value;
            settings.Training.Advanced.UserThresholdPriorVariance = this.ThresholdPriorVariance.Value;

            settings.Prediction.SetPredictionLossFunction(this.RatingLossFunction.Value);
        }
    }
}