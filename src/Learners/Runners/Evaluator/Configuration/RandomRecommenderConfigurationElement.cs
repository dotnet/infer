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
    /// The configuration element that describes the random recommender.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "RandomRecommender")]
    public class RandomRecommenderConfigurationElement : RecommenderConfigurationElement
    {
        /// <summary>
        /// Creates an instance of the random recommender.
        /// </summary>
        /// <param name="mapping">The mapping for the recommender being created.</param>
        /// <returns>An instance of the random recommender.</returns>
        public override IRecommender<SplitInstanceSource<RecommenderDataset>, User, Item, int, RatingDistribution, DummyFeatureSource> Create(
            IStarRatingRecommenderMapping<SplitInstanceSource<RecommenderDataset>, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping)
        {
            return new RandomStarRatingRecommender<SplitInstanceSource<RecommenderDataset>, RatedUserItem, User, Item, int, DummyFeatureSource, Vector>(mapping);
        }
    }
}