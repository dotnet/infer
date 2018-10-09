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
    /// The configuration element that describes the Mahout recommender.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "MahoutRecommender")]
    public class MahoutRecommenderConfigurationElement : RecommenderConfigurationElement
    {
        /// <summary>
        /// Gets or sets the rating similarity function used by the Mahout recommender.
        /// </summary>
        public MahoutRatingSimilarity? RatingSimilarity { get; set; }

        /// <summary>
        /// Gets or sets the algorithm used for rating prediction by the Mahout recommender.
        /// </summary>
        public MahoutRatingPredictionAlgorithm? RatingPredictionAlgorithm { get; set; }

        /// <summary>
        /// Gets or sets the algorithm used to fill the ratings that the Mahout recommender failed to estimate.
        /// </summary>
        public MahoutMissingRatingPredictionAlgorithm? MissingRatingPredictionAlgorithm { get; set; }

        /// <summary>
        /// Gets or sets the size of the user neighborhood used by the user-based rating prediction algorithm.
        /// </summary>
        public int? UserNeighborhoodSize { get; set; }

        /// <summary>
        /// Gets or sets the number of traits.
        /// </summary>
        public int? TraitCount { get; set; }

        /// <summary>
        /// Gets or sets the number of iterations.
        /// </summary>
        public int? IterationCount { get; set; }


        /// <summary>
        /// Gets or sets a value indicating whether to use 64-bit JVM to run Mahout
        /// </summary>
        public bool? UseX64JVM { get; set; }

        /// <summary>
        /// Gets or sets maximum heap size in MB for JVM running Mahout
        /// </summary>
        public int? JavaMaxHeapSizeInMb { get; set; }

        /// <summary>
        /// Creates an instance of the Mahout recommender using the settings from this configuration element.
        /// </summary>
        /// <param name="mapping">The mapping for the recommender being created.</param>
        /// <returns>An instance of the Mahout recommender.</returns>
        public override IRecommender<SplitInstanceSource<RecommenderDataset>, User, Item, int, RatingDistribution, DummyFeatureSource> Create(
            IStarRatingRecommenderMapping<SplitInstanceSource<RecommenderDataset>, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping)
        {
            var recommender = MahoutRecommender<SplitInstanceSource<RecommenderDataset>>.Create(mapping);
            
            recommender.Settings.RatingSimilarity = this.RatingSimilarity.Value;
            recommender.Settings.RatingPredictionAlgorithm = this.RatingPredictionAlgorithm.Value;
            recommender.Settings.MissingRatingPredictionAlgorithm = this.MissingRatingPredictionAlgorithm.Value;
            recommender.Settings.UserNeighborhoodSize = this.UserNeighborhoodSize.Value;
            recommender.Settings.TraitCount = this.TraitCount.Value;
            recommender.Settings.IterationCount = this.IterationCount.Value;
            recommender.Settings.UseX64JVM = this.UseX64JVM.Value;
            recommender.Settings.JavaMaxHeapSizeInMb = this.JavaMaxHeapSizeInMb.Value;
            
            return recommender;
        }
    }
}