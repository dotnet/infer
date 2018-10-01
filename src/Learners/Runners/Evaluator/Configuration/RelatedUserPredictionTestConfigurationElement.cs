// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Xml.Serialization;

    /// <summary>
    /// The configuration element that describes the test for related user prediction.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "PredictRelatedUsers")]
    public class RelatedUserPredictionTestConfigurationElement : RecommenderTestConfigurationElement
    {
        /// <summary>
        /// Gets or sets the maximum number of users to relate to a user during the test.
        /// </summary>
        public int? MaxRelatedUserCount { get; set; }

        /// <summary>
        /// Gets or sets the minimum number of common ratings between two users.
        /// </summary>
        public int? MinCommonRatingCount { get; set; }

        /// <summary>
        /// Gets or sets the minimum size of the related user pool for a single user.
        /// </summary>
        public int? MinRelatedUserPoolSize { get; set; }
        
        /// <summary>
        /// Creates an instance of <see cref="RelatedUserPredictionTestConfigurationElement"/> using the settings from this configuration element.
        /// </summary>
        /// <returns>An instance of <see cref="RelatedUserPredictionTestConfigurationElement"/>.</returns>
        public override RecommenderTest Create()
        {
            return new RelatedUserPredictionTest(this.MaxRelatedUserCount.Value, this.MinCommonRatingCount.Value, this.MinRelatedUserPoolSize.Value);
        }
    }
}