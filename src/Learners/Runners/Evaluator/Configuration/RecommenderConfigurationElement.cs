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
    /// The base class for all the configuration elements describing various recommenders.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "Recommender")]
    [XmlInclude(typeof(MatchboxRecommenderConfigurationElement))]
    [XmlInclude(typeof(VowpalWabbitRecommenderConfigurationElement))]
    [XmlInclude(typeof(MahoutRecommenderConfigurationElement))]
    [XmlInclude(typeof(RandomRecommenderConfigurationElement))]
    public abstract class RecommenderConfigurationElement : ConfigurationElement
    {
        /// <summary>
        /// Creates an instance of the recommender described by this configuration element.
        /// </summary>
        /// <param name="mapping">The mapping for the recommender being created.</param>
        /// <returns>The created recommender instance.</returns>
        public abstract IRecommender<SplitInstanceSource<RecommenderDataset>, User, Item, int, RatingDistribution, DummyFeatureSource> Create(
            IStarRatingRecommenderMapping<SplitInstanceSource<RecommenderDataset>, RatedUserItem, User, Item, int, DummyFeatureSource, Vector> mapping);
    }
}