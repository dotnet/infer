// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Xml.Serialization;

    using Microsoft.ML.Probabilistic.Learners.Mappings;
    using Microsoft.ML.Probabilistic.Math;

    /// <summary>
    /// The configuration element that describes splitting settings for recommendation datasets.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "RecommenderDatasetSplitSettings")]
    public class RecommenderDatasetSplitSettingsConfigurationElement : ConfigurationElement
    {
        /// <summary>
        /// Gets or sets the number of folds.
        /// </summary>
        public int? FoldCount { get; set; }
        
        /// <summary>
        ///  Gets or sets the fraction of users included in the training set only.
        /// </summary>
        public double? TrainingOnlyUserFraction { get; set; }

        /// <summary>
        /// Gets or sets the fraction of ratings in the training set for each user who is presented in both sets.
        /// </summary>
        public double? TestUserRatingTrainingFraction { get; set; }

        /// <summary>
        /// Gets or sets the fraction of users included in the test set only.
        /// </summary>
        public double? ColdUserFraction { get; set; }

        /// <summary>
        /// Gets or sets the fraction of items included in the test set only.
        /// </summary>
        public double? ColdItemFraction { get; set; }

        /// <summary>
        /// Gets or sets the fraction of users not included in either training or test set.
        /// </summary>
        public double? IgnoredUserFraction { get; set; }

        /// <summary>
        /// Gets or sets the fraction of items not included in either training or test set.
        /// </summary>
        public double? IgnoredItemFraction { get; set; }

        /// <summary>
        /// Gets or sets whether the occasionally produced cold items should be removed from the test set.
        /// </summary>
        public bool? RemoveOccasionalColdItems { get; set; }

        /// <summary>
        /// Creates an instance of <see cref="TrainTestSplittingStarRatingRecommenderMapping{TInstanceSource, TInstance, TUser, TItem, TRating, TFeatureSource, TFeatureValues}"/>
        /// described by this configuration element.
        /// </summary>
        /// <returns>The created splitting mapping.</returns>
        public TrainTestSplittingStarRatingRecommenderMapping<RecommenderDataset, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> CreateSplittingMapping()
        {
            return Mappings.StarRatingRecommender.SplitToTrainTest(
                this.TrainingOnlyUserFraction.Value,
                this.TestUserRatingTrainingFraction.Value,
                this.ColdUserFraction.Value,
                this.ColdItemFraction.Value,
                this.IgnoredUserFraction.Value,
                this.IgnoredItemFraction.Value,
                this.RemoveOccasionalColdItems.Value);
        }
    }
}