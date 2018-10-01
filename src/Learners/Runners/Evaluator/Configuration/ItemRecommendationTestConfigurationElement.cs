// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

namespace Microsoft.ML.Probabilistic.Learners.Runners
{
    using System;
    using System.Xml.Serialization;

    /// <summary>
    /// The configuration element that describes the item recommendation test.
    /// </summary>
    [Serializable]
    [XmlType(TypeName = "RecommendItems")]
    public class ItemRecommendationTestConfigurationElement : RecommenderTestConfigurationElement
    {
        /// <summary>
        /// Gets or sets the number of items to recommend to a user during the test.
        /// </summary>
        public int? MaxRecommendedItemCount { get; set; }

        /// <summary>
        /// Gets or sets the minimum size of the recommendation pool.
        /// </summary>
        public int? MinRecommendationPoolSize { get; set; }
        
        /// <summary>
        /// Creates an instance of <see cref="ItemRecommendationTest"/> using the settings from this configuration element.
        /// </summary>
        /// <returns>An instance of <see cref="ItemRecommendationTest"/>.</returns>
        public override RecommenderTest Create()
        {
            return new ItemRecommendationTest(this.MaxRecommendedItemCount.Value, this.MinRecommendationPoolSize.Value);
        }
    }
}