// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Xml.Serialization;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    using RatingDistribution = System.Collections.Generic.IDictionary<int, double>;

    /// <summary>
    /// The configuration element that describes the VW recommender.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "VWRecommender")]
    public class VowpalWabbitRecommenderConfigurationElement : RecommenderConfigurationElement
    {
        /// <summary>
        /// Gets or sets the number of traits.
        /// </summary>
        public int? TraitCount { get; set; }

        /// <summary>
        /// Gets or sets the number of bits in the feature table.
        /// </summary>
        public int? BitPrecision { get; set; }

        /// <summary>
        /// Gets or sets the learning rate.
        /// </summary>
        public double? LearningRate { get; set; }

        /// <summary>
        /// Gets or sets the learning rate decay.
        /// </summary>
        public double? LearningRateDecay { get; set; }

        /// <summary>
        /// Gets or sets the number of passes.
        /// </summary>
        public int? PassCount { get; set; }

        /// <summary>
        /// Gets or sets the weight of the L1 regularization term.
        /// </summary>
        public double? L1Regularization { get; set; }

        /// <summary>
        /// Gets or sets the weight of the L2 regularization term.
        /// </summary>
        public double? L2Regularization { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether user features should be used.
        /// </summary>
        public bool? UseUserFeatures { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether item features should be used.
        /// </summary>
        public bool? UseItemFeatures { get; set; }

        /// <summary>
        /// Creates an instance of the VW recommender using the settings from this configuration element.
        /// </summary>
        /// <param name="mapping">The mapping for the recommender being created.</param>
        /// <returns>An instance of the VW recommender.</returns>
        public override IRecommender<SplitInstanceSource<RecommenderDataset>, User, Item, int, RatingDistribution, DummyFeatureSource> Create(
            IStarRatingRecommenderMapping<SplitInstanceSource<RecommenderDataset>, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping)
        {
            var recommender = VowpalWabbitRecommender<SplitInstanceSource<RecommenderDataset>>.Create(mapping);

            recommender.Settings.TraitCount = this.TraitCount.Value;
            recommender.Settings.BitPrecision = this.BitPrecision.Value;
            recommender.Settings.LearningRate = this.LearningRate.Value;
            recommender.Settings.LearningRateDecay = this.LearningRateDecay.Value;
            recommender.Settings.PassCount = this.PassCount.Value;
            recommender.Settings.L1Regularization = this.L1Regularization.Value;
            recommender.Settings.L2Regularization = this.L2Regularization.Value;
            recommender.Settings.UseUserFeatures = this.UseUserFeatures.Value;
            recommender.Settings.UseItemFeatures = this.UseItemFeatures.Value;

            return recommender;
        }
    }
}