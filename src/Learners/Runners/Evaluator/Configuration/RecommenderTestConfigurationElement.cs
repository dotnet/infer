// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Xml.Serialization;

    /// <summary>
    /// The base class for all the configuration elements describing various recommender tests.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "Test")]
    [XmlInclude(typeof(RatingPredictionTestConfigurationElement))]
    [XmlInclude(typeof(ItemRecommendationTestConfigurationElement))]
    [XmlInclude(typeof(RelatedUserPredictionTestConfigurationElement))]
    [XmlInclude(typeof(RelatedItemPredictionTestConfigurationElement))]
    public abstract class RecommenderTestConfigurationElement : ConfigurationElement
    {
        /// <summary>
        /// Creates an instance of recommender test described by this configuration element.
        /// </summary>
        /// <returns>An instance of recommender test.</returns>
        public abstract RecommenderTest Create();
    }
}